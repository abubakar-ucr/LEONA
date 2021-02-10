import tempfile
import argparse
import os

from overrides import overrides

import torch
import torch.optim as optim
import numpy as np

from torch.nn.modules.linear import Linear

import allennlp

from allennlp.common.checks import check_dimensions_match, ConfigurationError

from allennlp.models import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.modules import ConditionalRandomField, FeedForward
from allennlp.modules.conditional_random_field import allowed_transitions


from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import sequence_cross_entropy_with_logits

from typing import Dict, Optional, Iterable, List, Tuple, Any, cast

from allennlp.data import DataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import SequenceLabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.samplers import BasicBatchSampler, BucketBatchSampler, SequentialSampler

from allennlp.training.metrics import CategoricalAccuracy


from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics.fbeta_measure import FBetaMeasure
from allennlp.training.metrics.span_based_f1_measure import SpanBasedF1Measure

import allennlp.nn.util as util
from allennlp.training.util import evaluate

from allennlp.modules.seq2vec_encoders import CnnEncoder

from allennlp.modules.token_embedders import (
    Embedding,
    TokenCharactersEncoder,
    ElmoTokenEmbedder,
    PretrainedTransformerEmbedder,
    PretrainedTransformerMismatchedEmbedder,
)

from allennlp.data.tokenizers import (
    CharacterTokenizer,
    PretrainedTransformerTokenizer,
    SpacyTokenizer,
    WhitespaceTokenizer,
)
from allennlp.data.token_indexers import (
    SingleIdTokenIndexer,
    TokenCharactersIndexer,
    ELMoTokenCharactersIndexer,
    PretrainedTransformerIndexer,
    PretrainedTransformerMismatchedIndexer,
)

from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import ReduceOnPlateauLearningRateScheduler

from allennlp.modules.matrix_attention import (
    LinearMatrixAttention, 
    MatrixAttention
)

from allennlp.nn import Activation

from allennlp.nn.util import replace_masked_values, min_value_of_dtype

from allennlp.nn import InitializerApplicator


def get_parameter():
    parser = argparse.ArgumentParser(description='Training E2E ZS slot filling model')

    parser.add_argument('-input_path', action="store", default="../data/")
    parser.add_argument('-out_path', action="store", default="../model_output/")
    parser.add_argument('-dataset', action="store", default="snips")

    args = parser.parse_args()
    return args

def replace_masked_values_with_big_negative_number(x: torch.Tensor, mask: torch.Tensor):
    return replace_masked_values(x, mask, min_value_of_dtype(x.dtype))


@DatasetReader.register('iob-tag')
class IOBDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer_space = WhitespaceTokenizer()
        self.tokenizer_spacy = SpacyTokenizer(language = "en_core_web_md", 
                                              pos_tags = True, split_on_spaces = True)
        self.token_indexers = {
            'elmo_tokens': ELMoTokenCharactersIndexer(),
            'token_characters': TokenCharactersIndexer(namespace='character_vocab',
                                                      min_padding_length=6),
            'pos_tags': SingleIdTokenIndexer(namespace='pos_tag_vocab',
                                     feature_name='tag_'),
            'ner_tags': SingleIdTokenIndexer(namespace='ner_tag_vocab',
                                     feature_name='ent_type_')
        } 
        
        self.slot_indexers = {
            'elmo_tokens': ELMoTokenCharactersIndexer(),
            'token_characters': TokenCharactersIndexer(namespace='character_vocab',
                                                      min_padding_length=6)
        }
        
        
    def text_to_instance(self, tokens: List[Token], slot: List[Token], 
                         s1_tags: List[str] = None,
                         tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        slot_field = TextField(slot, self.slot_indexers)
        
        fields = {"sentence": sentence_field, 
                  "slot": slot_field
                 }

        if s1_tags:
            s1_field = SequenceLabelField(labels=s1_tags, sequence_field=sentence_field,
                                         label_namespace = "s1_labels")
            fields["s1_labels"] = s1_field
        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field
        
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for line in f:
                sentence, s1_label, description, tags = line.strip().split('\t')
                yield self.text_to_instance(self.tokenizer_spacy.tokenize(sentence),
                                            self.tokenizer_spacy.tokenize(description),
                                            [iob for iob in s1_label.split()],
                                            [iob for iob in tags.split()])




class CrfTagger(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        utterance_embedder: TextFieldEmbedder,
        utterance_embedder2: TextFieldEmbedder,
        slot_embedder: TextFieldEmbedder,
        utterance_encoder: Seq2SeqEncoder,
        utterance_encoder2: Seq2SeqEncoder,
        slot_encoder: Seq2SeqEncoder,
        matrix_attention: MatrixAttention,
        modeling_layer: Seq2SeqEncoder,
        fc_ff_layer = FeedForward,
        label_namespace: str = "labels",
        s1_label_namespace: str = "s1_labels",
        feedforward: Optional[FeedForward] = None,
        label_encoding: Optional[str] = "BIO",
        include_start_end_transitions: bool = True,
        constrain_crf_decoding: bool = True,
        calculate_span_f1: bool = True,
        dropout: Optional[float] = 0.3,
        mask_lstms: bool = True,
        verbose_metrics: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        top_k: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.label_namespace = label_namespace
        self.s1_label_namespace = s1_label_namespace
        
        self.utterance_embedder = utterance_embedder
        self.utterance_embedder2 = utterance_embedder2
        self.slot_embedder = slot_embedder
        
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.s1_num_tags = self.vocab.get_vocab_size(s1_label_namespace)
        
        self.utterance_encoder = utterance_encoder
        self.utterance_encoder2 = utterance_encoder2
        self.slot_encoder = slot_encoder
        
        self._matrix_attention = matrix_attention
        self._modeling_layer = modeling_layer
        self.fc_ff_layer = fc_ff_layer
        self.top_k = top_k
        
        self._verbose_metrics = verbose_metrics
        
        self._mask_lstms = mask_lstms
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
            s1_output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.fc_ff_layer.get_output_dim()
            s1_output_dim = self.utterance_encoder.get_output_dim()
            
        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_tags))
        self.s1_tag_projection_layer = TimeDistributed(Linear(s1_output_dim, self.s1_num_tags))


        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError(
                    "constrain_crf_decoding is True, but no label_encoding was specified."
                )
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            s1_labels = self.vocab.get_index_to_token_vocabulary(s1_label_namespace)
            
            constraints = allowed_transitions(label_encoding, labels)
            s1_constraints = allowed_transitions(label_encoding, s1_labels)
        else:
            constraints = None
            s1_constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
            self.num_tags, constraints, include_start_end_transitions=include_start_end_transitions
        )
        
        #s1
        self.s1_crf = ConditionalRandomField(
            self.s1_num_tags, constraints, include_start_end_transitions=include_start_end_transitions
        )
        
        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError(
                    "calculate_span_f1 is True, but no label_encoding was specified."
                )
            self._f1_metric = SpanBasedF1Measure(
                vocab, tag_namespace=label_namespace, label_encoding=label_encoding
            )
            
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": self._f1_metric
        }

        check_dimensions_match(
            utterance_embedder.get_output_dim(),
            utterance_encoder.get_input_dim(),
            "utterance field embedding dim",
            "utterance encoder input dim",
        )
        if feedforward is not None:
            check_dimensions_match(
                modeling_layer.get_output_dim(),
                feedforward.get_input_dim(),
                "encoder output dim",
                "feedforward input dim",
            )
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        sentence: Dict[str, Dict[str, torch.Tensor]],#tokens: TextFieldTensors,
        slot: Dict[str, Dict[str, torch.Tensor]],
        s1_labels: torch.LongTensor = None, #s1_tags:
        labels: torch.LongTensor = None, #tags:
        metadata: List[Dict[str, Any]] = None,
        ignore_loss_on_o_tags: bool = False,
        **kwargs,  # to allow for a more general dataset reader that passes args we don't need
    ) -> Dict[str, torch.Tensor]:

        mask_sen = util.get_text_field_mask(sentence)
        mask_slot = util.get_text_field_mask(slot)
        
        slot_lstm_mask = mask_slot if self._mask_lstms else None
        utterance_lstm_mask = mask_sen if self._mask_lstms else None
        
        embedded_utterance = self.utterance_embedder(sentence)
        embedded_slot = self.slot_embedder(slot)
        embedded_utterance2 = self.utterance_embedder2(sentence)
        
        batch_size = embedded_slot.size(0)
        utterance_length = embedded_utterance.size(1)

        if self.dropout:
            embedded_utterance = self.dropout(embedded_utterance)
            embedded_slot = self.dropout(embedded_slot)
            embedded_utterance2 = self.dropout(embedded_utterance2)

        encoded_utterance = self.utterance_encoder(embedded_utterance, utterance_lstm_mask)
        encoded_slot = self.slot_encoder(embedded_slot, slot_lstm_mask)
        encoded_utterance2 = self.utterance_encoder2(embedded_utterance2, utterance_lstm_mask)

        if self.dropout:
            encoded_utterance = self.dropout(encoded_utterance)
            encoded_slot = self.dropout(encoded_slot)
            encoded_utterance2 = self.dropout(encoded_utterance2)
        
        encoding_dim_slot = encoded_slot.size(-1)
        
        #attention
        # Shape: (batch_size, passage_length, question_length)
        utterance_slot_similarity = self._matrix_attention(encoded_utterance, encoded_slot)
        # Shape: (batch_size, passage_length, question_length)
        utterance_slot_attention = util.masked_softmax(utterance_slot_similarity, mask_slot)
        # Shape: (batch_size, passage_length, encoding_dim)
        utterance_slot_vectors = util.weighted_sum(encoded_slot, utterance_slot_attention)

        # We replace masked values with something really negative here, so they don't affect the
        # max below.
        masked_similarity = replace_masked_values_with_big_negative_number(
            utterance_slot_similarity, mask_slot.unsqueeze(1)
        )
        # Shape: (batch_size, passage_length)
        slot_utterance_similarity = masked_similarity.max(dim=-1)[0].squeeze(-1)
        # Shape: (batch_size, passage_length)
        slot_utterance_attention = util.masked_softmax(slot_utterance_similarity, mask_sen)
        # Shape: (batch_size, encoding_dim)
        slot_utterance_vector = util.weighted_sum(encoded_utterance, slot_utterance_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        tiled_slot_utterance_vector = slot_utterance_vector.unsqueeze(1).expand(
            batch_size, utterance_length, encoding_dim_slot
        )

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        final_merged_utterance = torch.cat(
            [
                encoded_utterance,
                utterance_slot_vectors,
                encoded_utterance * utterance_slot_vectors,
                encoded_utterance * tiled_slot_utterance_vector,
                encoded_utterance2, # encoding for IOB prediction
            ],
            dim=-1,
        )
        
        modeled_utterance = self._modeling_layer(final_merged_utterance, utterance_lstm_mask)
        
        if self.dropout:
            modeled_utterance = self.dropout(modeled_utterance)
            
        ##end
        
        ## FF layer
        joined_modeled_utterance_s1 = self.fc_ff_layer(modeled_utterance)
        
        if self._feedforward is not None:
            joined_modeled_utterance_s1 = self._feedforward(joined_modeled_utterance_s1)

        logits = self.tag_projection_layer(joined_modeled_utterance_s1)
        best_paths = self.crf.viterbi_tags(logits, mask_sen, top_k=self.top_k)

        # Just get the top tags and ignore the scores.
        predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])

        output = {"logits": logits, "mask": mask_sen, "tags": predicted_tags}

        #s1
        s1_logits = self.s1_tag_projection_layer(encoded_utterance2)
        s1_best_paths = self.s1_crf.viterbi_tags(s1_logits, mask_sen, top_k=self.top_k)

        # Just get the top tags and ignore the scores.
        s1_predicted_tags = cast(List[List[int]], [x[0][0] for x in s1_best_paths])

        output["logits"] = s1_logits
        output["s1_tags"] = s1_predicted_tags
        #end s1
        
        if self.top_k > 1:
            output["top_k_tags"] = best_paths
            output["s1_top_k_tags"] = s1_best_paths

        if labels is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.label_namespace)
                crf_mask = mask_sen & (labels != o_tag_index)
            else:
                crf_mask = mask_sen
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, labels, crf_mask)
            s1_log_likelihood = self.s1_crf(s1_logits, s1_labels, crf_mask)

            loss = log_likelihood + s1_log_likelihood
            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, labels, mask_sen)
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, labels, mask_sen)
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        def decode_tags(tags):
            return [
                self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in tags
            ]

        def decode_top_k_tags(top_k_tags):
            return [
                {"tags": decode_tags(scored_path[0]), "score": scored_path[1]}
                for scored_path in top_k_tags
            ]

        output_dict["tags"] = [decode_tags(t) for t in output_dict["tags"]]

        if "top_k_tags" in output_dict:
            output_dict["top_k_tags"] = [decode_top_k_tags(t) for t in output_dict["top_k_tags"]]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self.calculate_span_f1:
            f1_scores = self.metrics["f1"].get_metric(reset=reset)
            return { "f1": f1_scores["f1-measure-overall"],
                    "acc.": self.metrics["accuracy"].get_metric(reset=reset)
                   }
        else:
            return { "acc.": self.metrics["accuracy"].get_metric(reset=reset)
                   }
            
    default_predictor = "sentence_tagger"


def build_dataset_reader() -> DatasetReader:
    return IOBDatasetReader()


def read_data(reader: DatasetReader, tgt_domain: str, input_path:str, domains:List) -> Tuple[Iterable[Instance], Iterable[Instance]]:
    print("Reading data")
    
    training_data = None
    for domain in domains:
        if domain != tgt_domain:
            if training_data == None:
                training_data = reader.read(input_path+domain+'/'+domain+'_neg.txt')
            else:
                training_data += reader.read(input_path+domain+'/'+domain+'_neg.txt')
            
    valid_test_data = reader.read(input_path+tgt_domain+'/'+tgt_domain+'_neg.txt')

    as_per_percent = int(len(valid_test_data) * 0.25) 
    valid_size = 2000 if as_per_percent >= 2000 else as_per_percent
    
    validation_data = valid_test_data[:valid_size]
    test_data = valid_test_data[valid_size:]
    
    training_data = AllennlpDataset(training_data)
    validation_data = AllennlpDataset(validation_data)
    test_data = AllennlpDataset(test_data)
    
    print("train:",len(training_data), "validation:", len(validation_data), "test:", len(test_data))
    return training_data, validation_data, test_data

def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300
    NUM_FILTERS = 60
    NGRAM_FILTER_SIZES = (2, 3, 4, 5, 6)
    #out_dim for char = len(NGRAM_FILTER_SIZES) * NUM_FILTERS
    F_OUT = 200
    
    
    elmo_options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    elmo_weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    
    elmo_embedding = ElmoTokenEmbedder(options_file=elmo_options_file,
                                   weight_file=elmo_weight_file)

    
    character_embedding = Embedding(vocab = vocab,
                                    embedding_dim = EMBEDDING_DIM,
                                    vocab_namespace = 'character_vocab'
                                )
    cnn_encoder = CnnEncoder(embedding_dim=EMBEDDING_DIM, 
                             num_filters=NUM_FILTERS, 
                             ngram_filter_sizes = NGRAM_FILTER_SIZES
                            )
    token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)


    pos_tag_embedding = Embedding(vocab=vocab, 
                                  embedding_dim=EMBEDDING_DIM,
                                  vocab_namespace='pos_tag_vocab'
                                 )
    
    ner_tag_embedding = Embedding(vocab=vocab, 
                                  embedding_dim=EMBEDDING_DIM,
                                  vocab_namespace='ner_tag_vocab'
                                 )
    
    word_embedding = Embedding(vocab = vocab,
                                    embedding_dim = EMBEDDING_DIM,
                                    vocab_namespace = 'token_vocab'
                                )
    

    utterance_embedder = BasicTextFieldEmbedder(token_embedders={'elmo_tokens': elmo_embedding,
                                                       'token_characters': token_encoder,
                                                       'pos_tags': pos_tag_embedding,
                                                      'ner_tags': ner_tag_embedding}
                                     )
    
    #slot embed
    slot_embedder = BasicTextFieldEmbedder(token_embedders={'elmo_tokens': elmo_embedding,
                                                       'token_characters': token_encoder,
                                                       }
                                     )
    
    utterance_lstm = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(2 * EMBEDDING_DIM + 1024 + len(NGRAM_FILTER_SIZES) * NUM_FILTERS,
                                               HIDDEN_DIM, 
                                               num_layers = 2,
                                               batch_first=True,
                                               bidirectional = True
                                               ))
    slot_lstm = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(1024 + len(NGRAM_FILTER_SIZES) * NUM_FILTERS,
                                               HIDDEN_DIM, 
                                               num_layers = 2,
                                               batch_first=True,
                                               bidirectional = True
                                               ))
    
    similarity = LinearMatrixAttention(tensor_1_dim=2*HIDDEN_DIM, 
                                       tensor_2_dim=2*HIDDEN_DIM,
                                       combination="x,y,x*y", 
                                       activation = Activation.by_name('tanh')())
    
    modeling_lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(2* 5 * HIDDEN_DIM, # bi-direction
                                                        HIDDEN_DIM,
                                                        num_layers = 2,
                                                        batch_first=True,
                                                        bidirectional = True
                                                       ))
    
    #step1_utterance
    utterance_embedder2 = BasicTextFieldEmbedder(token_embedders={'elmo_tokens': elmo_embedding,
                                                       'token_characters': token_encoder,
                                                       'pos_tags': pos_tag_embedding,
                                                      'ner_tags': ner_tag_embedding}
                                     )
    utterance_lstm2 = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(2 * EMBEDDING_DIM + 1024 + len(NGRAM_FILTER_SIZES) * NUM_FILTERS,
                                               HIDDEN_DIM, 
                                               num_layers = 2,
                                               batch_first=True,
                                               bidirectional = True
                                               ))
    
    ## FF to combines two lstm inputs
    final_linear_layer = FeedForward(2 * HIDDEN_DIM, 
                                2, 
                                [HIDDEN_DIM, F_OUT], 
                                torch.nn.ReLU(),
                                0.3
                                )
    #CRF model
    model = CrfTagger(vocab = vocab,
                     utterance_embedder = utterance_embedder, 
                      utterance_embedder2 = utterance_embedder2, 
                      slot_embedder = slot_embedder,
                      utterance_encoder = utterance_lstm, 
                      utterance_encoder2 = utterance_lstm2, 
                      slot_encoder = slot_lstm,
                      matrix_attention = similarity,
                      modeling_layer = modeling_lstm,
                      fc_ff_layer = final_linear_layer
                     )
    return model



def build_data_loaders(train_data: torch.utils.data.Dataset,
                       dev_data: torch.utils.data.Dataset,
                      ) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=32, shuffle=False)
    return train_loader, dev_loader



def build_trainer(model: Model, serialization_dir: str, train_loader: DataLoader,
                  dev_loader: DataLoader) -> Trainer:

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("cpu/gpu? ",device)    

    model = model.to(device)
    
    parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr = 0.001)
    scheduler = ReduceOnPlateauLearningRateScheduler(optimizer = optimizer,
                                                     patience = 5,
                                                    verbose=True)

    trainer = GradientDescentTrainer(model=model, serialization_dir=serialization_dir, cuda_device = device, 
                                     data_loader=train_loader, validation_data_loader=dev_loader, 
                                     learning_rate_scheduler = scheduler,
                                     patience=20, num_epochs=200,
                                     optimizer=optimizer,
                                    validation_metric = "+f1",
                                    )
    return trainer


def run_training_loop(target_domain, all_domains, input_path, out_path):
    dataset_reader = build_dataset_reader()

    train_data, dev_data, test_data = read_data(dataset_reader, target_domain, input_path, all_domains)

    vocab = build_vocab(train_data + dev_data + test_data)
    print(vocab)
    model = build_model(vocab)
   

    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    test_data.index_with(vocab)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)


    serialization_dir = out_path+target_domain+"/"
    
    trainer = build_trainer(
            model,
            serialization_dir,
            train_loader,
            dev_loader
        )
    trainer.train()
        
    #save vocab
    vocab.save_to_files(serialization_dir+"vocabulary")
    
    #save the model.
    with open(serialization_dir+"model_zs.pt", 'wb') as f:
        torch.save(model.state_dict(), f)
    
    return model, dataset_reader, test_data

def get_domains(path):
    all_intents = [x[0].replace(path, "") for x in os.walk(path) if len(x[0].replace(path, "") ) > 0 
           and x[0].replace(path, "")[0]!="."]
    return all_intents

if __name__ == '__main__':
    args = get_parameter()
    input_path = args.input_path
    out_path = args.out_path
    dataset = args.dataset


    domains = get_domains(input_path+dataset+"/")

    for domain in domains:
        print("starting", domain)

        model, dataset_reader, test_data = run_training_loop(domain, domains, input_path+dataset+"/", out_path)
        test_data_loader = DataLoader(test_data, batch_size=32)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("cpu/gpu? ", device) 

        results = evaluate(model, test_data_loader, cuda_device = device)
        print("results for", domain)
        print(results)


