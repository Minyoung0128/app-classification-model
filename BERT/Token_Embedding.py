import sentencepiece as spm


templates= '--input={} \
--model_prefix={} \
--pad_id={} \
--bos_id={} \
--eos_id={} \
--unk_id={} \
--pad_piece={} \
--bos_piece={} \
--eos_piece={} \
--unk_piece={} \
--vocab_size={} \
--character_coverage={} \
--model_type={} \
--max_sentence_length={}'


train_input_file = "packet_text.txt"
vocab_size = 65536 # vocab 사이즈
prefix = 'packet_tokenizer_bpe' # 저장될 tokenizer 모델에 붙는 이름

pad_id=0  #<pad> token을 0으로 설정
bos_id=1 #<start> token을 1으로 설정
eos_id=2 #<end> token을 2으로 설정
unk_id=3 #<unknown> token을 3으로 설정

pad_piece = '[PAD]'
bos_piece = '[CLS]'
eos_piece = '[END]'
unk_piece = '[UNK]'

character_coverage = 1.0 # to reduce character set 
model_type ='bpe' # Choose from unigram (default), bpe, char, or word

max_sentence_length = 80

cmd = templates.format(train_input_file, prefix,
                       pad_id, bos_id, eos_id, unk_id,
                       pad_piece, bos_piece, eos_piece, unk_piece,
                       vocab_size, character_coverage, model_type, max_sentence_length
                       )

print(cmd)
spm.SentencePieceTrainer.Train(cmd)

