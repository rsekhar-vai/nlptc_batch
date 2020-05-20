from torch.utils.data import Dataset
import re
from sklearn.model_selection import train_test_split
import spacy
 


class TextDataset(Dataset):
    def __init__(self, args):        
			text_df_orig = pd.read_csv(args.text_csv, encoding='unicode_escape')
			f text_column != 'text':
				text_df_orig.rename(columns={args.text_column: 'text'}, inplace=True)
			if label_column != 'category':
					text_df_orig.rename(columns={args.label_column: 'category'}, inplace=True)
			self.full = text_df_orig[['text', 'category']]

			self._max_seq_length = args.max_text_length
			train_temp, self.test = train_test_split(self.full)
			self.train, self.validation = train_test_split(train_temp)

			self.train_size = len(self.train)
			self.validation_size = len(self.validation)
			self.test_size = len(self.test)

			train_df.to_csv(args.train_csv, index=False)
			val_df.to_csv(args.val_csv, index=False)
			test_df.to_csv(args.test_csv, index=False)

    def get_splits(self):
			return self.train, self.validation, self.test


class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X, y)


class Tokenizer:
    def __init__(self, token_type='w'):
        if token_type == 'c':
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
