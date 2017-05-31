#import gensim
# https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
#m = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, encoding='utf8')
#print(m.most_similar('king'))
import fasttext
model = fasttext.load_model('wiki.en.bin', label_prefix='__label__')
print(dir(model), type(model), model.model_name)
fasttext.supervised('a', 'b', pretrained_vectors='wiki.en.vec')
#print(model['king'])
#print(model.predict(['it was dark outside']))
