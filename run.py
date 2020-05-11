from argparse import Namespace

import pandas as pd
import torch.optim as optim
from torchtext import data
from torchtext import vocab

from models import *
from utilfunctions import *

if __name__ == '__main__':
    args = Namespace(
        text_csv=None,
        train_csv="data/train.csv",
        val_csv="data/val.csv",
        test_csv="data/test.csv",
        model_state_file="model.pth",
        save_dir="model_storage/Clf",
        glove_filepath='D:\\Projects\\Text Analytics\\Glove\\glove.6B.100d.txt',
        hidden_dim=100,
        num_channels=100,
        seed=1337,
        learning_rate=0.001,
        dropout_p=0.1,
        batch_size=64,
        num_epochs=20,
        early_stopping_criteria=5,
        cuda=True,
        catch_keyboard_interrupt=True,
        reload_from_files=False,
        expand_filepaths_to_save_dir=True,
        token_type='w',
        max_text_length=256,
        pretrained_embeddings='Glove',
        embedding_size=100,
        build_simple_char_cnn=False,
        build_simple_word_cnn=True,
        build_convrec_bilstm=False,
        build_vdcnn=False,

    )

    setup_environment(args)

    text_column = 'text'
    label_column = 'category'
    args.text_csv = "data/bbcheadlines_text.csv"
    text_df_orig = pd.read_csv(args.text_csv, encoding='unicode_escape')
    print(text_df_orig.columns)

    if text_column != 'text':
        text_df_orig.rename(columns={text_column: 'text'}, inplace=True)
    if label_column != 'category':
        text_df_orig.rename(columns={label_column: 'category'}, inplace=True)
    text_df = text_df_orig[['text', 'category']]
    print(text_df.columns)

    dataset = TextDataset(text_df, args)
    train_df, val_df, test_df = dataset.get_splits()

    train_df.to_csv(args.train_csv, index=False)
    val_df.to_csv(args.val_csv, index=False)
    test_df.to_csv(args.test_csv, index=False)

    if args.build_simple_word_cnn:
        print("=================== Build Simple Word CNN ===================")
        max_text_length = 256,
        args.token_type = 'w'
        tokenizer = Tokenizer(args.token_type).tokenizer
        print(tokenizer('this is test#Q! Z&*!'))
        Field_TEXT = data.Field(tokenize=tokenizer, sequential=True,
                                use_vocab=True, batch_first=True, fix_length=args.max_text_length)
        Field_LABEL = data.LabelField(sequential=False)
        mapping_with_file_columns = [('text', Field_TEXT), ('category', Field_LABEL)]
        Dataset_train, Dataset_val, Dataset_test = data.TabularDataset.splits(
            path='',
            train=args.train_csv,
            validation=args.val_csv,
            test=args.test_csv,
            format='csv',
            fields=mapping_with_file_columns,
            skip_header=True
        )
        print(vars(Dataset_train[0]))

    skip = True
    if not skip:
        Field_TEXT = data.Field(tokenize=tokenizer, sequential=True, use_vocab=True)
        Field_LABEL = data.LabelField(sequential=False)
        mapping_with_file_columns = [('text', Field_TEXT), ('category', Field_LABEL)]
        Dataset_train, Dataset_val = data.TabularDataset.splits(
            path='',
            train=args.train_csv,
            validation=args.val_csv,
            format='csv',
            fields=mapping_with_file_columns,
            skip_header=True
        )

        MAX_VOCAB_SIZE = 25000
        vec = vocab.Vectors('glove.6B.100d.txt', 'Glove')
        Field_TEXT.build_vocab(Dataset_train, Dataset_val,
                               max_size=MAX_VOCAB_SIZE,
                               #                              vectors="glove.6B.100d",
                               vectors=vec,
                               unk_init=torch.Tensor.normal_)
        Field_LABEL.build_vocab(Dataset_train)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        Batches_train, Batches_val = data.BucketIterator.splits(
            (Dataset_train, Dataset_val),
            batch_size=args.batch_size,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
            device=device)

        # Use GloVe or randomly initialized embeddings
        embeddings = Field_TEXT.vocab.vectors.numpy()

        classifier = WordCNN_Simple(embedding_size=args.embedding_size,
                                    num_embeddings=len(Field_TEXT.vocab),
                                    num_channels=args.num_channels,
                                    hidden_dim=args.hidden_dim,
                                    num_classes=len(Field_LABEL.vocab),
                                    dropout_p=args.dropout_p,
                                    pretrained_embeddings=embeddings,
                                    padding_idx=0)
        classifier = classifier.to(args.device)
        # dataset.class_weights = dataset.class_weights.to(args.device)
        # loss_func = nn.CrossEntropyLoss(dataset.class_weights)
        loss_func = nn.CrossEntropyLoss()

        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)
        print("------- # of Parameters ---->: ", sum(p.numel() for p in classifier.parameters() if p.requires_grad))

        results = build_model(args, dataset, classifier, loss_func, optimizer, scheduler)

        print(results)

    if args.build_vdcnn:
        print("======================== Build Very Deep CNN ================================")
        args.max_text_length = 300
        args.token_type = 'c'
        args.num_channels = 256

        dataset, vectorizer = load_vectorize_data(args)
        if args.pretrained_embeddings == 'Glove':
            words = vectorizer.text_vocab._token_to_idx.keys()
            embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
                                               words=words)
            print("Using Glove embeddings")
        else:
            print("Not using pre-trained embeddings")
            embeddings = None
        classifier = VDCNN(num_classes=len(vectorizer.category_vocab),
                           embedding_dim=args.embedding_size,
                           k_max=8, embedding_size=len(vectorizer.text_vocab))
        classifier = classifier.to(args.device)
        dataset.class_weights = dataset.class_weights.to(args.device)
        loss_func = nn.CrossEntropyLoss(dataset.class_weights)
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)
        print("------- # of Parameters ---->: ", sum(p.numel() for p in classifier.parameters() if p.requires_grad))

        results = build_model(args, dataset, classifier, loss_func, optimizer, scheduler)
        print(results)

    if args.build_convrec_bilstm:
        print("======================== Build CNN + RNN/BiLSTM ================================")
        args.max_text_length = 300
        args.token_type = 'c'
        args.num_channels = 256

        dataset, vectorizer = load_vectorize_data(args)
        embeddings = None
        classifier = ConvRec_BiLSTM(embedding_size=args.embedding_size,
                                    num_embeddings=len(vectorizer.text_vocab),
                                    num_channels=args.num_channels,
                                    hidden_dim=args.hidden_dim,
                                    num_classes=len(vectorizer.category_vocab),
                                    dropout_p=args.dropout_p,
                                    pretrained_embeddings=None,
                                    padding_idx=0)
        classifier = classifier.to(args.device)
        dataset.class_weights = dataset.class_weights.to(args.device)
        loss_func = nn.CrossEntropyLoss(dataset.class_weights)
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)
        print("------- # of Parameters ---->: ", sum(p.numel() for p in classifier.parameters() if p.requires_grad))

        results = build_model(args, dataset, classifier, loss_func, optimizer, scheduler)
        print(results)

    if args.build_simple_char_cnn:
        print("======================== Build Simple Char CNN ================================")
        args.max_text_length = 300
        args.token_type = 'c'
        args.num_channels = 256

        dataset, vectorizer = load_vectorize_data(args)
        embeddings = None
        classifier = CharCNN_Simple(embedding_size=args.embedding_size,
                                    num_embeddings=len(vectorizer.text_vocab),
                                    num_channels=args.num_channels,
                                    hidden_dim=args.hidden_dim,
                                    num_classes=len(vectorizer.category_vocab),
                                    dropout_p=args.dropout_p,
                                    pretrained_embeddings=None,
                                    padding_idx=0)
        classifier = classifier.to(args.device)
        dataset.class_weights = dataset.class_weights.to(args.device)
        loss_func = nn.CrossEntropyLoss(dataset.class_weights)
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)
        print("------- # of Parameters ---->: ", sum(p.numel() for p in classifier.parameters() if p.requires_grad))

        results = build_model(args, dataset, classifier, loss_func, optimizer, scheduler)
        print(results)
