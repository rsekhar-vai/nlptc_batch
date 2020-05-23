from argparse import Namespace

import torch.optim as optim
from torchtext import vocab

from models import *
from utilclasses import *
from utilfunctions import *

if __name__ == '__main__':
    args = Namespace(
        # text_csv=None, # updated subsequently
        train_csv="data/train.csv",
        val_csv="data/val.csv",
        test_csv="data/test.csv",
        model_state_file="model.pth",
        save_dir="model_storage/Clf",
        glove_filepath='D:\\Projects\\Glove',
        hidden_dim=100,
        num_channels=100,
        seed=1337,
        learning_rate=0.001,
        dropout_p=0.1,
        batch_size=16,
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
        build_simple_cnn=True,
        build_vdcnn=False
    )
    setup_environment(args)

    args.text_column = 'text'
    args.label_column = 'category'
    args.text_csv = "data/bbcheadlines_text.csv"
    dataset = TextDataset(args)

    if args.build_simple_cnn:
        print("=================== Build Simple Word CNN ===================")
        args.token_type = 'w'
        args.max_text_length = 256
        args.tokenizer = Tokenizer(args.token_type).tokenizer
        args.word_vectors = vocab.Vectors('glove.6B.100d.txt', args.glove_filepath)
        # args.word_vectors = []
        pp = PreProcessor(args)
        classifier = SimpleCNN(embedding_size=args.embedding_size,
                               num_embeddings=pp.get_text_vocab_length(),
                               num_channels=args.num_channels,
                               hidden_dim=args.hidden_dim,
                               num_classes=pp.get_label_vocab_length(),
                               dropout_p=args.dropout_p,
                               pretrained_embeddings=pp.get_embeddings(),
                               padding_idx=0)
        classifier = classifier.to(args.device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)
        print("------- # of Parameters ---->: ", sum(p.numel() for p in classifier.parameters() if p.requires_grad))
        results = build_model(args, pp, classifier, loss_func, optimizer, scheduler)
        print(results)

    if args.build_vdcnn:
        print("======================== Build Very Deep CNN ================================")
        args.token_type = 'w'
        args.max_text_length = 128
        args.tokenizer = Tokenizer(args.token_type).tokenizer
        args.word_vectors = vocab.Vectors('glove.6B.100d.txt', args.glove_filepath)
        pp = PreProcessor(args)
        classifier = VDCNN(num_classes=pp.get_label_vocab_length(),
                           embedding_dim=args.embedding_size,
                           k_max=8, embedding_size=args.embedding_size)
        classifier = classifier.to(args.device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                         mode='min', factor=0.5,
                                                         patience=1)
        print("------- # of Parameters ---->: ", sum(p.numel() for p in classifier.parameters() if p.requires_grad))
        results = build_model(args, pp, classifier, loss_func, optimizer, scheduler)
        print(results)
