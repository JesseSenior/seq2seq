import random
import re

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar

from dataset import Lang, normalizeString, SOS_TOKEN, EOS_TOKEN


class TrainerProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.bar = None
        self.enabled = True

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        if self.enabled:
            self.bar = tqdm(total=trainer.max_epochs, initial=trainer.current_epoch)
            self.running_loss = 0.0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.bar:
            postfix = self.get_metrics(trainer, pl_module)
            for key in postfix.keys():
                if type(postfix[key]) == float:
                    postfix[key] = "%.3f" % postfix[key]
            self.bar.set_postfix(postfix)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.bar:
            self.bar.update(1)

    def disable(self):
        self.bar = None
        self.enabled = False


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Model(pl.LightningModule):
    def __init__(self, input_lang: Lang, output_lang: Lang, **kwargs) -> None:
        super().__init__()
        self.input_lang = input_lang
        self.output_lang = output_lang

    def infer(self, input_str: str) -> str:
        pass


class Seq2seq(Model):
    def __init__(
        self,
        max_length: int,
        hidden_size: int = 256,
        dropout_prob: float = 0.1,
        learning_rate=0.01,
        teacher_forcing_ratio=0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.criterion = nn.NLLLoss()

        self.encoder = EncoderRNN(self.input_lang.n_words, hidden_size)
        self.decoder = AttnDecoderRNN(
            hidden_size,
            self.output_lang.n_words,
            self.max_length,
            dropout_p=dropout_prob,
        )

        self.history = {"train_loss": [], "val_loss": []}
        self.save_hyperparameters()

    def on_train_epoch_start(self) -> None:
        self.train_loss_list = []

    def training_step(self, batch, batch_idx):
        input_sentences, target_sentences = batch
        loss = 0
        loss_len = 0
        for input_sentence, target_sentence in zip(input_sentences, target_sentences):
            encoder_hidden = self.encoder.initHidden().to(self.device)

            input_tensor = self.input_lang.toTensor(input_sentence).to(self.device)
            target_tensor = self.output_lang.toTensor(target_sentence).to(self.device)

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)
            loss_len += target_length

            encoder_outputs = torch.zeros(
                self.max_length, self.hidden_size, device=self.device
            )

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[ei], encoder_hidden
                )
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_TOKEN]], device=self.device)

            decoder_hidden = encoder_hidden

            use_teacher_forcing = random.random() < self.teacher_forcing_ratio

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, _ = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    loss += self.criterion(decoder_output, target_tensor[di])
                    decoder_input = target_tensor[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, _ = self.decoder(
                        decoder_input, decoder_hidden, encoder_outputs
                    )
                    _, topi = decoder_output.topk(1)
                    decoder_input = (
                        topi.squeeze().detach()
                    )  # detach from history as input

                    loss += self.criterion(decoder_output, target_tensor[di])
                    if decoder_input.item() == EOS_TOKEN:
                        break
        loss /= loss_len
        self.train_loss_list.append(loss)
        self.log("loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        try:
            train_loss = sum(self.train_loss_list) / len(self.train_loss_list)
            self.history["train_loss"].append(train_loss)
        except ZeroDivisionError:
            pass

    def on_validation_epoch_start(self) -> None:
        self.val_loss_list = []

    def validation_step(self, batch, batch_idx):
        input_sentences, target_sentences = batch
        loss = 0
        loss_len = 0
        for input_sentence, target_sentence in zip(input_sentences, target_sentences):
            encoder_hidden = self.encoder.initHidden().to(self.device)

            input_tensor = self.input_lang.toTensor(input_sentence).to(self.device)
            target_tensor = self.output_lang.toTensor(target_sentence).to(self.device)

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)
            loss_len += target_length

            encoder_outputs = torch.zeros(
                self.max_length, self.hidden_size, device=self.device
            )

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(
                    input_tensor[ei], encoder_hidden
                )
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_TOKEN]], device=self.device)

            decoder_hidden = encoder_hidden

            for di in range(target_length):
                decoder_output, decoder_hidden, _ = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_TOKEN:
                    break

        loss /= loss_len
        self.val_loss_list.append(loss)
        return loss

    def on_validation_epoch_end(self):
        val_loss = sum(self.val_loss_list).item() / len(self.val_loss_list)
        self.history["val_loss"].append(val_loss)
        self.log("val_loss", val_loss, prog_bar=True)
        try:
            tqdm.write(
                f"[Train] Loss: {self.history['train_loss'][-1]:.3f} Val Loss: {self.history['val_loss'][-1]:.3f}"
            )
        except IndexError:
            pass

    def forward(self, sentence):
        input_tensor = self.input_lang.toTensor(sentence).to(self.device)
        input_length = input_tensor.size()[0]
        encoder_hidden = self.encoder.initHidden().to(self.device)

        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=self.device
        )

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden
            )
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=self.device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_tensor = []
        decoder_attentions = torch.zeros(
            self.max_length, self.max_length, device=self.device
        )

        for di in range(self.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            decoded_tensor.append(topi.item())
            if topi.item() == EOS_TOKEN:
                break

            decoder_input = topi.squeeze().detach()
        decoded_tensor = torch.tensor(decoded_tensor)
        return self.output_lang.toSentence(decoded_tensor)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def infer(self, input_str: str):
        input_str = normalizeString(input_str)
        output_str = self.forward(input_str)
        if self.output_lang.name == "cmn":
            return output_str.replace(" ", "").replace("<EOS>", "")
        return (
            re.sub(r"\s+([.,;!?])", r"\1", output_str).replace("<EOS>", "").capitalize()
        )
