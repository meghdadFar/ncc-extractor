import sys
reload(sys)
sys.setdefaultencoding('utf8')
from path import add_parent_to_path
add_parent_to_path()
from functions import read_corpus, replace_compounds, reddy_ncs, pre_process, print_every, read_ncs, \
    write_to_file, get_preprocess_args
import logging
from config import logging_config
import nltk
nltk.download('punkt')
nltk.download('wordnet')


if __name__ == '__main__':
    logging.info('Reading train and evaluation ncs')
    args = get_preprocess_args()
    r_ncs, _ = reddy_ncs(args.p2ec)
    ncs = read_ncs(args.p2tc)
    ncs.extend(r_ncs)
    logging.info('Reading corpus')
    sentences = read_corpus(args.p2corp)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    output = []
    logging.info('Replacing ncs in corpus')
    for i in range(0, len(sentences)):
        s = sentences[i]
        print_every(s, i+1, 10000)
        output.append(replace_compounds(pre_process(s, lemmatizer), ncs))
    logging.info('Writing results in ' + args.p2out)
    write_to_file(output, args.p2out)