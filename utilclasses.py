import re

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchtext import data


class TextDataset(Dataset):
    def __init__(self, args):
        text_df_orig = pd.read_csv(args.text_csv, encoding='unicode_escape')
        if args.text_column != 'text':
            text_df_orig.rename(columns={args.text_column: 'text'}, inplace=True)
        if args.label_column != 'category':
            text_df_orig.rename(columns={args.label_column: 'category'}, inplace=True)
        self.full_df = text_df_orig[['text', 'category']]

        self._max_seq_length = args.max_text_length
        train_temp, self.test_df = train_test_split(self.full_df)
        self.train_df, self.validation_df = train_test_split(train_temp)

        self.train_size = len(self.train_df)
        self.validation_size = len(self.validation_df)
        self.test_size = len(self.test_df)

        self.train_df.to_csv(args.train_csv, index=False)
        self.validation_df.to_csv(args.val_csv, index=False)
        self.test_df.to_csv(args.test_csv, index=False)

    def get_splits(self):
        return self.train_df, self.validation_df, self.test_df


class Tokenizer:
    def __init__(self, args):
        if args.token_type == 'c':
            self.tokenizer = Tokenizer.char_tokenizer
        else:
            self.tokenizer = Tokenizer.word_tokenizer()

    @staticmethod
    def word_tokenizer():
        nlp = spacy.load('en_core_web_sm', disable=["tagger", "parser", "ner"])

        def tokenizer(sentence):
            tokens = [w.text.lower() for w in nlp(Tokenizer.clean_sentence(sentence))]
            return tokens

        return tokenizer

    @staticmethod
    def char_tokenizer(sentence):
        tokens = [w.lower() for w in Tokenizer.clean_sentence(sentence)]
        return tokens

    @staticmethod
    def clean_sentence(text):
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # remove non alphanumeric character
        return text.strip()


class PreProcessor:
    def __init__(self, args):
        self.token_type = args.token_type
        self.Field_TEXT = data.Field(tokenize=args.tokenizer, sequential=True,
                                     use_vocab=True, batch_first=True, fix_length=args.max_text_length)
        self.Field_LABEL = data.LabelField(sequential=False)
        mapping_with_file_columns = [('text', self.Field_TEXT), ('category', self.Field_LABEL)]
        self.Dataset_train, self.Dataset_val, self.Dataset_test = data.TabularDataset.splits(
            path='',
            train=args.train_csv,
            validation=args.val_csv,
            test=args.test_csv,
            format='csv',
            fields=mapping_with_file_columns,
            skip_header=True
        )
        MAX_VOCAB_SIZE = 25000
        if args.word_vectors is None:
            args.word_vectors = []
        self.Field_TEXT.build_vocab(self.Dataset_train, self.Dataset_val,
                                    max_size=MAX_VOCAB_SIZE,
                                    vectors=args.word_vectors)
        self.Field_LABEL.build_vocab(self.Dataset_train)

    def get_text_vocab(self):
        return self.Field_TEXT.vocab.stoi

    def get_text_vocab_length(self):
        return len(self.Field_TEXT.vocab)

    def get_text_vocab_vector(self, token):
        return self.Field_TEXT.vocab.vectors[self.Field_TEXT.vocab.stoi[token]]

    def get_label_vocab(self):
        return self.Field_LABEL.vocab.stoi

    def get_label_vocab_length(self):
        return len(self.Field_LABEL.vocab)

    def get_embeddings(self):
        if self.token_type == 'c':
            return None
        elif self.token_type == 'w':
            return self.Field_TEXT.vocab.vectors.numpy()
        else:
            return self.Field_TEXT.vocab.vectors.numpy()

    def idxtosent(self, batch, idx):
        return ' '.join([self.Field_TEXT.vocab.itos[i] for i in batch.text[idx, :].cpu().data.numpy()])


class BatchGenerator:
    def __init__(self, args, ds):
        self.dl = data.BucketIterator(
            (ds),
            batch_size=args.batch_size,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=args.device)
        self.x_field = 'text'
        self.y_field = 'category'

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield X, y
