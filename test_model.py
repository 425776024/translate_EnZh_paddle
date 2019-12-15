import numpy as np
import pandas as pd
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers

bos_id = 0  # 词典中start token对应的id
eos_id = 1  # 词典中end token对应的id

# 原数据（en_vocab.txt）词汇表大小
source_dict_size = 7152
# 翻译对象数据（zh_vocab.txt）词汇表大小
target_dict_size = 13833

word_dim = 512  # 词向量维度
hidden_dim = 512  # 编码器中的隐层大小
decoder_size = hidden_dim  # 解码器中的隐层大小
max_length = 256  # 解码生成句子的最大长度
beam_size = 4  # beam search的柱宽度
batch_size = 64  # batch 中的样本数
# 训练数据
data_path = 'data.csv'
# 模型保存路径
model_save_dir = "machine_translation.model"


def data_func(is_train=True):
    # 源语言source数据
    src = fluid.data(name="src", shape=[None, None], dtype="int64")
    src_sequence_length = fluid.data(name="src_sequence_length",
                                     shape=[None],
                                     dtype="int64")
    inputs = [src, src_sequence_length]
    # 训练时还需要目标语言target和label数据
    if is_train:
        trg = fluid.data(name="trg", shape=[None, None], dtype="int64")
        trg_sequence_length = fluid.data(name="trg_sequence_length",
                                         shape=[None],
                                         dtype="int64")
        label = fluid.data(name="label", shape=[None, None], dtype="int64")
        inputs += [trg, trg_sequence_length, label]
    # data loader
    loader = fluid.io.DataLoader.from_generator(feed_list=inputs,
                                                capacity=10,
                                                iterable=True,
                                                use_double_buffer=True)
    return inputs, loader


def encoder(src_embedding, src_sequence_length):
    #  src_embedding:[batch_size, sequence_length, ...]
    #  双向GRU编码器

    # 使用GRUCell构建前向RNN
    encoder_fwd_cell = layers.GRUCell(hidden_size=hidden_dim)
    encoder_fwd_output, fwd_state = layers.rnn(
        cell=encoder_fwd_cell,
        inputs=src_embedding,
        sequence_length=src_sequence_length,
        time_major=False,
        is_reverse=False)

    # 使用GRUCell构建反向RNN
    encoder_bwd_cell = layers.GRUCell(hidden_size=hidden_dim)
    encoder_bwd_output, bwd_state = layers.rnn(
        cell=encoder_bwd_cell,
        inputs=src_embedding,
        sequence_length=src_sequence_length,
        time_major=False,
        is_reverse=True)

    # 拼接前向与反向GRU的编码结果得到h
    encoder_output = layers.concat(
        input=[encoder_fwd_output, encoder_bwd_output], axis=2)
    encoder_state = layers.concat(input=[fwd_state, bwd_state], axis=1)
    return encoder_output, encoder_state


class DecoderCell(layers.RNNCell):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.gru_cell = layers.GRUCell(hidden_size)

    def attention(self, hidden, encoder_output, encoder_output_proj, encoder_padding_mask):
        # 定义attention用以计算context，即 c_i，这里使用Bahdanau attention机制
        decoder_state_proj = layers.unsqueeze(layers.fc(hidden, size=self.hidden_size, bias_attr=False), [1])
        mixed_state = fluid.layers.elementwise_add(encoder_output_proj, layers.expand(decoder_state_proj, [1,
                                                                                                           layers.shape(
                                                                                                               decoder_state_proj)[
                                                                                                               1], 1]))
        attn_scores = layers.squeeze(layers.fc(input=mixed_state, size=1, num_flatten_dims=2, bias_attr=False), [2])
        if encoder_padding_mask is not None:
            attn_scores = layers.elementwise_add(attn_scores, encoder_padding_mask)
        attn_scores = layers.softmax(attn_scores)
        context = layers.reduce_sum(layers.elementwise_mul(encoder_output, attn_scores, axis=0), dim=1)
        return context

    def call(self, step_input, hidden, encoder_output, encoder_output_proj, encoder_padding_mask=None):
        # Bahdanau attention
        context = self.attention(hidden, encoder_output, encoder_output_proj, encoder_padding_mask)
        step_input = layers.concat([step_input, context], axis=1)
        # GRU
        output, new_hidden = self.gru_cell(step_input, hidden)
        return output, new_hidden


def decoder(encoder_output, encoder_output_proj, encoder_state,
            encoder_padding_mask, trg=None, is_train=True):
    # 定义 RNN 所需要的组件
    decoder_cell = DecoderCell(hidden_size=decoder_size)
    decoder_initial_states = layers.fc(encoder_state, size=decoder_size, act="tanh")
    trg_embeder = lambda x: fluid.embedding(input=x, size=[target_dict_size, hidden_dim],
                                            dtype="float32",
                                            param_attr=fluid.ParamAttr(name="trg_emb_table"))

    output_layer = lambda x: layers.fc(input=x,
                                       size=target_dict_size,
                                       num_flatten_dims=len(x.shape) - 1,
                                       param_attr=fluid.ParamAttr(name="output_w"))
    if is_train:  # 训练
        # 训练时使用 `layers.rnn` 构造由 `cell` 指定的循环神经网络
        # 循环的每一步从 `inputs` 中切片产生输入，并执行 `cell.call`
        # [-1,-1,512,] , [-1,512,]
        decoder_output, _ = layers.rnn(
            cell=decoder_cell,
            inputs=trg_embeder(trg),
            initial_states=decoder_initial_states,
            time_major=False,
            encoder_output=encoder_output,
            encoder_output_proj=encoder_output_proj,
            encoder_padding_mask=encoder_padding_mask)

        decoder_output = layers.fc(input=decoder_output,
                                   size=target_dict_size,
                                   num_flatten_dims=2,
                                   param_attr=fluid.ParamAttr(name="output_w"))

    else:  # 基于 beam search 的预测生成
        # beam search 时需要将用到的形为 `[batch_size, ...]` 的张量扩展为 `[batch_size* beam_size, ...]`
        encoder_output = layers.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_output, beam_size)
        encoder_output_proj = layers.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_output_proj, beam_size)
        encoder_padding_mask = layers.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_padding_mask, beam_size)
        # BeamSearchDecoder 定义了单步解码的操作：`cell.call` + `beam_search_step`
        beam_search_decoder = layers.BeamSearchDecoder(cell=decoder_cell,
                                                       start_token=bos_id,
                                                       end_token=eos_id,
                                                       beam_size=beam_size,
                                                       embedding_fn=trg_embeder,
                                                       output_fn=output_layer)
        # 使用 layers.dynamic_decode 动态解码
        # 重复执行 `decoder.step()` 直到其返回的表示完成状态的张量中的值全部为True或解码步骤达到 `max_step_num`
        decoder_output, _ = layers.dynamic_decode(
            decoder=beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=max_length,
            output_time_major=False,
            encoder_output=encoder_output,
            encoder_output_proj=encoder_output_proj,
            encoder_padding_mask=encoder_padding_mask)

    return decoder_output


def model_func(inputs, is_train=True):
    # inputs = [src, src_sequence_length, trg, trg_sequence_length, label]
    # src = fluid.data(name="src", shape=[None, None], dtype="int64")
    # 源语言输入
    src = inputs[0]
    src_sequence_length = inputs[1]
    src_embedding = fluid.embedding(
        input=src,
        size=[source_dict_size, hidden_dim],
        dtype="float32",
        param_attr=fluid.ParamAttr(name="src_emb_table"))

    # 编码器
    encoder_output, encoder_state = encoder(src_embedding, src_sequence_length)

    encoder_output_proj = layers.fc(input=encoder_output,
                                    size=decoder_size,
                                    num_flatten_dims=2,
                                    bias_attr=False)
    src_mask = layers.sequence_mask(src_sequence_length,
                                    maxlen=layers.shape(src)[1],
                                    dtype="float32")
    encoder_padding_mask = (src_mask - 1.0) * 1e9

    # 目标语言输入，训练时有、预测生成时无该输入
    trg = inputs[2] if is_train else None

    # 解码器
    output = decoder(encoder_output=encoder_output,
                     encoder_output_proj=encoder_output_proj,
                     encoder_state=encoder_state,
                     encoder_padding_mask=encoder_padding_mask,
                     trg=trg,
                     is_train=is_train)
    return output


def load_vocab(path):
    vocab = {}
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            vocab[line] = i
    return vocab


def make_ids(words, vocab_dict):
    # 给输入词组转成id序列
    ids = []
    try:
        for w in words:
            ids.append(vocab_dict[w])
    except Exception:
        ids.append(vocab_dict['<unk>'])
    return ids


# 翻译准备的数据
en_vocab_path = 'en_vocab.txt'
zh_vocab_path = 'zh_vocab.txt'
en_word_ids = load_vocab(en_vocab_path)
zh_word_ids = load_vocab(zh_vocab_path)

src_idx2word = {v: k for k, v in en_word_ids.items()}
trg_idx2word = {v: k for k, v in zh_word_ids.items()}

# 开始预测，加载空模型
infer_prog = fluid.Program()
startup_prog = fluid.Program()
with fluid.program_guard(infer_prog, startup_prog):
    with fluid.unique_name.guard():
        inputs, loader = data_func(is_train=False)
        predict_seqs = model_func(inputs, is_train=False)

use_cuda = False
# 设置训练设备
place = fluid.CPUPlace()

# 定义执行器，加载model_save_dir的参数并绑定Program
exe = fluid.Executor(place)
exe.run(startup_prog)
fluid.io.load_params(exe, model_save_dir, main_program=infer_prog)
prog = fluid.CompiledProgram(infer_prog).with_data_parallel()

# 开始预测'I'm so happy'这句话，输入你得加上开头<s>，结尾<e>
test_data = ['<s>', 'I\'m', 'so', 'happy', '<e>']
test_token = make_ids(test_data, en_word_ids)

# 准备输入数据形式
feed_data = {'src': np.array([test_token]), 'src_sequence_length': np.array([len(test_token)])}
seq_ids = exe.run(prog, feed=feed_data, fetch_list=[predict_seqs])
# 以数组形式返回
seq_ids = seq_ids[0]
# 打印原数据，和beamsearch搜索的beam_size个最佳概率输出翻译
for ins_idx in range(seq_ids.shape[0]):
    print("英文:")
    src_seqs = np.array(feed_data["src"])
    print(" ".join([
        src_idx2word[idx] for idx in src_seqs[ins_idx][1:]
        if idx != eos_id
    ]))

    print("中文:")
    for beam_idx in range(beam_size):
        seq = [
            trg_idx2word[idx] for idx in seq_ids[ins_idx, :, beam_idx]
            if idx != eos_id
        ]
        print(" ".join(seq))
