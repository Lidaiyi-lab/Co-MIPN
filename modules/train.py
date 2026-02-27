import torch
from torch import optim
from tqdm import tqdm
import random
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import classification_report  # Standard metric for NER sequence labeling
from transformers.optimization import get_linear_schedule_with_warmup

from .metrics import eval_result


# --------------------------- Base Abstract Trainer Class ---------------------------
class BaseTrainer(object):
    """
    Abstract base class for all trainers (RE/NER)
    Defines core interfaces: train/evaluate/test (must be implemented by subclasses)
    """

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()


# --------------------------- Relation Extraction (RE) Trainer (Secondary) ---------------------------
class RETrainer(BaseTrainer):
    """
    Trainer for Relation Extraction (RE) task (secondary to NER)
    Handles RE model training, evaluation, and testing with standard classification metrics
    """

    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, args=None,
                 logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.re_dict = processor.get_relation_dict()  # RE label mapping (relation name -> id)
        self.logger = logger
        self.writer = writer  # TensorBoard writer for logging metrics
        self.refresh_step = 2  # Step interval for loss logging
        self.best_dev_metric = 0  # Track best dev F1 score for RE
        self.best_test_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs  # Total training steps
        self.step = 0
        self.args = args
        # Initialize optimizer/scheduler based on multimodal prompt usage
        if self.args.use_prompt:
            self.before_multimodal_train()
        else:
            self.before_train()

    def train(self):
        """
        Core training loop for RE task:
        1. Initialize training state and logging
        2. Load pre-trained model if specified
        3. Iterate over epochs/batches, compute loss, update parameters
        4. Evaluate on dev set after specified start epoch
        5. Track best performance and save model
        """
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        # Load pre-trained model weights if path provided
        if self.args.load_path is not None:
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        # Progress bar setup for training visualization
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    # Move batch tensors to target device (CPU/GPU)
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels = self._step(batch, mode="train")
                    avg_loss += loss.detach().cpu().item()

                    # Backpropagation and optimizer step
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Log training loss at specified refresh interval
                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)
                        avg_loss = 0

                # Evaluate on dev set after evaluation start epoch
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)

            pbar.close()
            self.pbar = None
            # Log best performance metrics
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                                    self.best_dev_metric))
            self.logger.info(
                "Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch,
                                                                                         self.best_test_metric))

    def evaluate(self, epoch):
        """
        RE evaluation on dev set:
        1. Switch model to eval mode (disable dropout/batch norm)
        2. Compute loss and predictions without gradient computation
        3. Calculate classification metrics (accuracy/micro-F1)
        4. Track best dev performance and save model checkpoint
        """
        self.model.eval()
        self.logger.info("***** Running evaluation *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        step = 0
        total_loss = 0

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")

                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    attention_mask, labels, logits, loss = self._step(batch, mode="dev")

                    total_loss += loss.detach().cpu().item()
                    avg_loss = total_loss / step

                    pbar.set_postfix_str(f"loss:{avg_loss:.5f}")
                    pbar.update()

                    # Convert logits to predictions (argmax for RE classification)
                    if isinstance(logits, torch.Tensor):
                        logits = logits.argmax(-1).detach().cpu().numpy()

                    # Process labels/predictions (filter padding/special tokens)
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}

                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[
                                    label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)

                pbar.close()

            # Calculate sequence labeling metrics (seqeval for RE)
            results = classification_report(y_true, y_pred, digits=4)
            self.logger.info("***** Dev Eval results *****")
            self.logger.info("\n%s", results)
            f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])
            avg_dev_loss = avg_loss

            # Log metrics to TensorBoard
            if self.writer:
                self.writer.add_scalar(tag='dev_f1', scalar_value=f1_score, global_step=epoch)
                self.writer.add_scalar(tag='dev_loss', scalar_value=avg_dev_loss, global_step=epoch)

            # Update best dev performance
            self.logger.info(
                "Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, current dev loss: {}." \
                    .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                            f1_score, avg_dev_loss))

            if f1_score >= self.best_dev_metric:
                self.logger.info("Get better performance at epoch {}".format(epoch))
                self.best_dev_epoch = epoch
                self.best_dev_metric = f1_score
                # Save best model checkpoint
                if self.args.save_path is not None:
                    torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                    self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self):
        """
        RE final testing on test set:
        1. Load best model checkpoint
        2. Evaluate on test set with same metrics as dev
        3. Log final test performance (accuracy/micro-F1)
        """
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels = self._step(batch, mode="dev")
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())

                    pbar.update()
                pbar.close()
                # Generate classification report for RE
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values())[1:],
                                                     target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc)
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1)
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss / len(self.test_data))
                total_loss = 0
                self.logger.info("Test f1 score: {}, acc: {}.".format(micro_f1, acc))

        self.model.train()

    def _step(self, batch, mode="train"):
        """
        Core forward step for RE:
        1. Unpack batch data (handle multimodal prompt data if enabled)
        2. Forward pass through model
        3. Return loss/logits and original labels
        """
        if mode != "predict":
            if self.args.use_prompt:
                input_ids, token_type_ids, attention_mask, labels, images, aux_imgs = batch
            else:
                images, aux_imgs = None, None
                input_ids, token_type_ids, attention_mask, labels = batch
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                 labels=labels, images=images, aux_imgs=aux_imgs)
            return outputs, labels

    def before_train(self):
        """
        Initialize optimizer/scheduler for standard (non-multimodal) RE training:
        1. Parameter grouping (no decay for bias/LayerNorm)
        2. AdamW optimizer with linear warmup scheduler
        3. Move model to target device
        """
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)

    def before_multimodal_train(self):
        """
        Initialize optimizer/scheduler for multimodal RE training:
        1. Separate parameter groups for BERT/visual prompt layers
        2. Freeze ResNet image model parameters (no gradient update)
        3. AdamW optimizer with linear warmup scheduler
        """
        optimizer_grouped_parameters = []
        # BERT parameters group
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        # Visual prompt parameters group (encoder_conv/gates)
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name:
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)

        # Freeze image model (ResNet) parameters
        for name, param in self.model.named_parameters():
            if 'image_model' in name:
                param.require_grad = False

        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)


# --------------------------- Named Entity Recognition (NER) Trainer (Primary) ---------------------------
class NERTrainer(BaseTrainer):
    """
    Core Trainer for Named Entity Recognition (NER) task (sequence labeling)
    Key features:
    - Supports both BERT-only and multimodal (text+vision) training
    - Uses seqeval metrics for NER evaluation (precision/recall/F1 for entity spans)
    - Separate learning rates for BERT/visual prompts/CRF layers
    - Tracks train/dev/test metrics and saves best model checkpoint
    """

    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, label_map=None,
                 args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.logger = logger
        self.label_map = label_map  # NER label mapping (label -> id, e.g., B-PER -> 1)
        self.writer = writer
        self.refresh_step = 2  # Step interval for loss logging
        self.best_dev_metric = 0  # Track best dev F1 score for NER
        self.best_test_metric = 0
        self.best_train_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.best_train_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs  # Total training steps
        self.step = 0
        self.args = args

    def train(self):
        """
        Core NER training loop:
        1. Initialize optimizer/scheduler (BERT-only or multimodal)
        2. Load pre-trained model if specified
        3. Iterate over epochs/batches:
           - Forward pass + loss computation
           - Backpropagation + parameter update
           - Log training loss/metrics
        4. Evaluate on dev set after specified start epoch
        5. Track best train/dev performance
        """
        # Initialize optimizer/scheduler based on multimodal prompt usage
        if self.args.use_prompt:
            self.multiModal_before_train()
        else:
            self.bert_before_train()

        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        # Load pre-trained model weights if path provided
        if self.args.load_path is not None:
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        # Progress bar setup for training visualization
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                y_true, y_pred = [], []
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    # Move batch tensors to target device (CPU/GPU)
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    attention_mask, labels, logits, loss = self._step(batch, mode="train")
                    avg_loss += loss.detach().cpu().item()

                    # Backpropagation and optimizer step
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Process NER predictions (handle CRF output format)
                    if isinstance(logits, torch.Tensor):  # CRF returns list for decoded sequences
                        logits = logits.argmax(-1).detach().cpu().numpy()
                    label_ids = labels.to('cpu').numpy()
                    input_mask = attention_mask.to('cpu').numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}

                    # Filter padding/special tokens (X/[SEP]) for NER evaluation
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[
                                    label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)

                    # Log training loss at specified refresh interval
                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)
                        avg_loss = 0

                # Calculate training metrics (seqeval F1 for NER)
                results = classification_report(y_true, y_pred, digits=4)
                self.logger.info("***** Train Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])
                if self.writer:
                    self.writer.add_scalar(tag='train_f1', scalar_value=f1_score, global_step=epoch)
                self.logger.info("Epoch {}/{}, best train f1: {}, best epoch: {}, current train f1 score: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_train_metric, self.best_train_epoch,
                                         f1_score))

                # Update best training performance
                if f1_score > self.best_train_metric:
                    self.best_train_metric = f1_score
                    self.best_train_epoch = epoch

                # Evaluate on dev set after evaluation start epoch
                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)

            torch.cuda.empty_cache()  # Release GPU memory
            pbar.close()
            self.pbar = None
            # Log best performance metrics
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                                    self.best_dev_metric))
            self.logger.info(
                "Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch,
                                                                                         self.best_test_metric))

    def evaluate(self, epoch):
        """
        NER evaluation on dev set:
        1. Switch model to eval mode (disable dropout/batch norm)
        2. Compute loss and predictions without gradient computation
        3. Calculate seqeval metrics (precision/recall/F1 for entity spans)
        4. Track best dev performance and save model checkpoint
        """
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        step = 0
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    attention_mask, labels, logits, loss = self._step(batch, mode="dev")
                    total_loss += loss.detach().cpu().item()

                    # Convert logits to predictions (handle CRF output format)
                    if isinstance(logits, torch.Tensor):
                        logits = logits.argmax(-1).detach().cpu().numpy()

                    # Process labels/predictions (filter padding/special tokens)
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[
                                    label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)

                    pbar.update()
                pbar.close()

                # Calculate seqeval metrics for NER (entity-level evaluation)
                results = classification_report(y_true, y_pred, digits=4)
                self.logger.info("***** Dev Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])

                # Log metrics to TensorBoard
                if self.writer:
                    self.writer.add_scalar(tag='dev_f1', scalar_value=f1_score, global_step=epoch)
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss / step, global_step=epoch)

                # Update best dev performance
                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                         f1_score))
                if f1_score >= self.best_dev_metric:
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = f1_score
                    # Save best model checkpoint
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self):
        """
        NER final testing on test set:
        1. Load best model checkpoint
        2. Evaluate on test set with seqeval metrics (entity-level F1)
        3. Log final test performance (F1 score for NER)
        """
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        y_true, y_pred = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    attention_mask, labels, logits, loss = self._step(batch, mode="dev")
                    total_loss += loss.detach().cpu().item()

                    # Convert logits to predictions (handle CRF output format)
                    if isinstance(logits, torch.Tensor):
                        logits = logits.argmax(-1).detach().cpu().tolist()

                    # Process labels/predictions (filter padding/special tokens)
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[
                                    label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)
                    pbar.update()
                pbar.close()

                # Calculate final seqeval metrics for test set
                results = classification_report(y_true, y_pred, digits=4)
                self.logger.info("***** Test Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])

                # Log metrics to TensorBoard
                if self.writer:
                    self.writer.add_scalar(tag='test_f1', scalar_value=f1_score)
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss / len(self.test_data))
                total_loss = 0
                self.logger.info("Test f1 score: {}.".format(f1_score))

        self.model.train()

    def _step(self, batch, mode="train"):
        """
        Core forward step for NER:
        1. Unpack batch data (handle multimodal prompt data if enabled)
        2. Forward pass through NER model (returns logits/loss)
        3. Return attention mask, labels, logits, and loss for metric calculation
        """
        if self.args.use_prompt:
            input_ids, token_type_ids, attention_mask, labels, images, aux_imgs = batch
        else:
            images, aux_imgs = None, None
            input_ids, token_type_ids, attention_mask, labels = batch

        # Forward pass through NER model (supports multimodal input)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels, images=images, aux_imgs=aux_imgs)
        logits, loss = output.logits, output.loss
        return attention_mask, labels, logits, loss

    def bert_before_train(self):
        """
        Initialize optimizer/scheduler for BERT-only NER training:
        1. AdamW optimizer with default learning rate
        2. Linear warmup scheduler
        3. Move model to target device (CPU/GPU)
        """
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.model.to(self.args.device)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)

    def multiModal_before_train(self):
        """
        Initialize optimizer/scheduler for multimodal NER training:
        1. Separate parameter groups with different learning rates:
           - BERT layers (base LR)
           - Visual prompt layers (encoder_conv/gates, base LR)
           - CRF/FC layers (higher LR for sequence labeling)
        2. Freeze image model (ResNet) parameters (no gradient update)
        3. AdamW optimizer with linear warmup scheduler
        4. Move model to target device (CPU/GPU)
        """
        parameters = []

        # BERT parameters group (base LR)
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                params['params'].append(param)
        parameters.append(params)

        # Visual prompt parameters group (encoder_conv/gates, base LR)
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name:
                params['params'].append(param)
        parameters.append(params)

        # CRF/FC parameters group (higher LR for sequence labeling)
        params = {'lr': 5e-2, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'crf' in name or name.startswith('fc'):
                params['params'].append(param)
        parameters.append(params)

        # Initialize AdamW optimizer with parameter groups
        self.optimizer = optim.AdamW(parameters)

        # Freeze image model (ResNet) parameters (no gradient update)
        for name, par in self.model.named_parameters():
            if 'image_model' in name:
                par.requires_grad = False

        # Linear warmup scheduler
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)