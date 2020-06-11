import os
import re
import sys
import time
import gzip
import argparse

import collections
import numpy as np

from IO_helper import load_corpus, load_embeddings

import spams

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

class CrossLingMapper:

    def __init__(self, lang_tgt, lang_src, preproc,
                 embedding_file_tgt, embedding_file_src,
                 dictionary_file=None, dict_fallback=False):
        assert lang_tgt == lang_src or embedding_file_tgt is not None # monolingual or bilingual with target language embeddings provided
        self.lang_T, self.lang_S = lang_tgt, lang_src
        self.embeddingsS, self.w2iS, self.i2wS = load_embeddings(embedding_file_src, max_words=500000)
        self.squared_normsS = np.sum(self.embeddingsS * self.embeddingsS, axis=1)
        self.perform_preproc_steps(preproc, self.embeddingsS)
        if embedding_file_tgt is None:
            self.i2wT = self.i2wS
        else:
            self.embeddingsT, self.w2iT, self.i2wT = load_embeddings(embedding_file_tgt, max_words=500000)
            self.squared_normsT = np.sum(self.embeddingsT * self.embeddingsT, axis=1)
            self.perform_preproc_steps(preproc, self.embeddingsT)
        self.dict_file = dictionary_file
        self.allow_dictionary_fallback = dict_fallback

    def perform_preproc_steps(self, preproc, embedding):
        for pp in preproc.split('-'):
            if pp == 'unit':
                self.length_normalize_rows(embedding)
            elif pp == 'center':
                self.mean_center_columns(embedding)

    def length_normalize_rows(self, embeddings):
        model_row_norms = np.sqrt((embeddings**2).sum(axis=1))[:, np.newaxis]
        embeddings /= model_row_norms
        return embeddings

    def mean_center_columns(self, embeddings):
        embeddings -= np.mean(embeddings, axis=0)
        return embeddings

    def length_normalize_columns(self, embeddings):
        model_col_norms = np.sqrt((embeddings**2).sum(axis=0))[np.newaxis,:]
        embeddings /= model_col_norms
        return embeddings

    def mean_center_rows(self, embeddings):
        embeddings -= np.mean(embeddings, axis=1)[:, np.newaxis]
        return embeddings


    def pseudo_align_embeddings(self):
        logging.warning("Note that the translation pairs are treated as identical surface form word pais.")
        mapped_embeddings_T, mapped_embeddings_S = [], []
        for w, iT in self.w2iT.items():
            if w in self.w2iS:
                mapped_embeddings_T.append(self.embeddingsT[iT])
                mapped_embeddings_S.append(self.embeddingsS[self.w2iS[w]])
                self.mapped_word_ids_T.append(iT)
                self.mapped_word_ids_S.append(self.w2iS[w])
        return mapped_embeddings_T, mapped_embeddings_S


    def align_embeddings(self, max_aligned=-1, reverse=False):
        mapped_embeddings_T, mapped_embeddings_S = [], []
        self.mapped_word_ids_T, self.mapped_word_ids_S = [], []
        use_pseudo_dictionary = self.dict_file is None or (self.allow_dictionary_fallback and not os.path.exists(self.dict_file))
        if use_pseudo_dictionary:
            mapped_embeddings_T, mapped_embeddings_S = self.pseudo_align_embeddings()
        else:
            distinct_words_seen = set()
            if self.dict_file.endswith('.gz'):
                f = gzip.open(self.dict_file, 'rt')
            else:
                f = open(self.dict_file)
            for line in f:
                splitted_line = re.split('( \|\|\| |\s+)', line.strip())
                distinct_words_seen.add(splitted_line[0])
                if len(distinct_words_seen) > max_aligned > 0:
                    break
                if reverse:
                    splitted_line = splitted_line[::-1]
                word_T = splitted_line[0].replace('{}:'.format(self.lang_T), '')
                word_S = splitted_line[-1].replace('{}:'.format(self.lang_S), '')

                if word_T in self.w2iT and word_S in self.w2iS:
                    word_T_id, word_S_id = self.w2iT[word_T], self.w2iS[word_S] # capitalization does matter
                    mapped_embeddings_T.append(self.embeddingsT[word_T_id])
                    mapped_embeddings_S.append(self.embeddingsS[word_S_id])
                    self.mapped_word_ids_T.append(word_T_id)
                    self.mapped_word_ids_S.append(word_S_id)
        logging.info('{} words aligned based on {}'.format(len(mapped_embeddings_T), self.dict_file))
        return np.array(mapped_embeddings_T), np.array(mapped_embeddings_S), use_pseudo_dictionary


    def transform_representations(self, tgt, src, mapping_mode='isometric'):
        target_trafo = None
        if self.lang_T != self.lang_S:
            target_trafo, source_trafo = self.determine_transformation(tgt, src, mapping_mode)
            self.embeddingsT_modded = self.embeddingsT @ target_trafo
        else:
            self.embeddingsT_modded = self.embeddingsT
        return target_trafo


    def determine_transformation(self, tgt, src, mapping_mode):
        source_trafo = None
        if mapping_mode == 'isometric':
            U, _, V = np.linalg.svd(src.T @ tgt)
            target_trafo = V.T @ U.T
        elif mapping_mode == 'pinv':
            target_trafo = np.linalg.pinv(tgt) @ src
        return target_trafo, source_trafo


    def learn_semantic_atoms(self, matrix, corpus_file, squared_norms, w2i, params, initial_D=None, file_id=None):
        if file_id is not None and os.path.exists('{}.dict.gz'.format(file_id)):
            D = np.loadtxt('{}.dict.gz'.format(file_id))
            return np.asfortranarray(D)

        D = np.asfortranarray(spams.trainDL(matrix.T, D=initial_D, **params))
        if file_id is not None:
            np.savetxt('{}.dict.gz'.format(file_id), D)
        return D


    def learn_sparse_coeffs(self, matrix, D, params, weighting=None):
        if weighting is not None:
            alphas = spams.lassoWeighted(matrix, D=D, W=weighting, **params)
        else:
            alphas = spams.lasso(matrix, D=D, **params)
        #logging.info('Alphas shape and sparsity:\t{}\t{:.4f}'.format(alphas.shape, 100*alphas.nnz/np.prod(alphas.shape)))
        return alphas


    def write_multiling_embeddings(self, embeddings, out_file_name):
        dense = type(embeddings) == np.ndarray
        dim = embeddings.shape[1 if dense else 0]
        language_prefix = '{}:'.format(self.lang_T) if self.lang_T is not None and len(self.lang_T) > 0 else ''
        with open(out_file_name, 'a') as f:
            for i in range(len(self.i2wT)):
                f.write('{}{}'.format(language_prefix, self.i2wT[i]))
                if dense:
                    to_print = embeddings[i]
                else:
                    c = embeddings.getcol(i)
                    to_print = collections.defaultdict(int, zip(c.indices, c.data))
                f.write(' {}\n'.format(' '.join(map(str, [round(to_print[j],8) for j in range(dim)]))))
                

def main():
    t = time.time()
    parser = argparse.ArgumentParser(description='Produces MaMuS (MAssively MUltilingual Sparse) word representations')
    parser.add_argument('--preproc-steps', help='in what way input vectors to be preprocessed', choices=['intact', 'unit', 'unit-center', 'center', 'center-unit'], required=False, default='unit')
    parser.add_argument('--embedding-mode', required=True)
    parser.add_argument('--lda', help='lambda for sparse coding [default: 0.1]', type=float, default=0.1)
    parser.add_argument('--K', help='number of basis vectors [default: 1200]', type=int, default=1200)
    parser.add_argument('--source-lang-id', help='lang ID to train on [default: en]', type=str, default='en')
    parser.add_argument('--target-lang-id', help='lang ID to evaluate on [default: fr]', type=str, default='fr')
    parser.add_argument('--source-embedding', help='source embeddings to read', type=str, required=True)
    parser.add_argument('--target-embedding', help='target embeddings to read', type=str, required=False)
    parser.add_argument('--dictionary-file', help='the dictionary file to use', type=str, required=False, default=None)
    parser.add_argument('--dictionary-fallback', help='Fallback policy in case of a missing dictionary file', action='store_true')
    parser.add_argument('--source-corpus', help='source background frequencies to obtain from', type=str, required=False, default=None)
    parser.add_argument('--target-corpus', help='target background frequencies to obtain from', type=str, required=False, default=None)
    parser.add_argument('--out-path', required=True)
    parser.add_argument('--trafo-type', choices=['isometric', 'pinv'], type=str, required=False, default='isometric')
    parser.add_argument('--max-aligned-words', type=int, required=False, default=-1)

    alphas_nonneg_parser = parser.add_mutually_exclusive_group(required=False)
    alphas_nonneg_parser.add_argument('--alphas-nonneg', dest='nonneg', action='store_true')
    alphas_nonneg_parser.add_argument('--alphas-any', dest='nonneg', action='store_false')
    parser.set_defaults(nonneg=True)

    args = parser.parse_args()

    if not os.path.exists('./decompositions'):
        os.mkdir('./decompositions')

    clm = CrossLingMapper(args.target_lang_id, args.source_lang_id, args.preproc_steps,
                          args.target_embedding, args.source_embedding,
                          args.dictionary_file, args.dictionary_fallback)
    logging.info(args)
    tgt, src, pseudo_dict = clm.align_embeddings(max_aligned=args.max_aligned_words)
    trafo = clm.transform_representations(tgt, src, mapping_mode=args.trafo_type)

    if args.lda > 0:
        params = {'K':args.K, 'lambda1':args.lda, 'numThreads':8, 'batchsize':400, 'iter':1000, 'verbose':False, 'posAlpha':args.nonneg}
        l_params = {x:params[x] for x in ['L','lambda1','lambda2','mode','pos','ols','numThreads','length_path','verbose'] if x in params}
        l_params['pos'] = args.nonneg
        fid = './decompositions/{}_{}_{}_{}_{}_{}'.format(args.source_lang_id,
                                                          'pos' if args.nonneg else 'nopos',
                                                          args.embedding_mode,
                                                          args.K,
                                                          args.lda,
                                                          args.preproc_steps)

        S_dict = clm.learn_semantic_atoms(clm.embeddingsS, args.source_corpus, clm.squared_normsS, clm.w2iS, params, file_id=fid)

        if args.source_lang_id == args.target_lang_id:
            S_alphas = clm.learn_sparse_coeffs(clm.embeddingsS.T, S_dict, l_params)

            nnz = 100*(1 - S_alphas.nnz / np.prod(S_alphas.shape))
            logging.info("time:\t{}\tnnz:\t{}".format(time.time() - t, nnz))
            clm.write_multiling_embeddings(S_alphas, args.out_path)
            sys.exit(1)

        T_alphas_modded = clm.learn_sparse_coeffs(clm.embeddingsT_modded.T, S_dict, l_params)

        clm.write_multiling_embeddings(T_alphas_modded, args.out_path)
        nnz = 100*(1 - T_alphas_modded.nnz / np.prod(T_alphas_modded.shape))
        logging.info("time:\t{}\tnnz:\t{}".format(time.time() - t, nnz))
    else: # do mapping directly based on the dense input emebeddings
        logging.info("The outputs are going to be dense embeddings as the command line argument for lambda was set to be zero.")
        clm.write_multiling_embeddings(clm.embeddingsT_modded, args.out_path)

if __name__ == "__main__":
  main()

