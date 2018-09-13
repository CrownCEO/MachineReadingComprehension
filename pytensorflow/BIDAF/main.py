import tensorflow as tf
import ujson as json
import numpy as np
import os
from tqdm import tqdm
from pytensorflow.BIDAF.model import Model
from pytensorflow.util import get_record_parser, get_batch_dataset, get_dataset, convert_tokens, evaluate


def train(config):
    # shape [91589, 300]
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    # shape [1427,64]
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)
    # {'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a
    # golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue
    # of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of
    #  the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It
    # is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette
    # Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the
    # Gold Dome), is a simple, modern stone statue of Mary.', 'spans': [[0, 15], [15, 16], [17, 20], [21, 27], [28,
    # 31], [32, 33], [34, 42], [43, 52], [52, 53], [54, 58], [59, 62], [63, 67], [68, 76], [76, 78], [79, 83], [84,
    # 88], [89, 91], [92, 93], [94, 100], [101, 107], [108, 110], [111, 114], [115, 121], [122, 126], [126, 127],
    # [128, 139], [140, 142], [143, 148], [149, 151], [152, 155], [156, 160], [161, 169], [170, 173], [174, 180],
    # [181, 183], [183, 184], [185, 187], [188, 189], [190, 196], [197, 203], [204, 206], [207, 213], [214, 218],
    # [219, 223], [224, 232], [233, 237], [238, 241], [242, 248], [249, 250], [250, 256], [257, 259], [260, 262],
    # [263, 268], [268, 269], [269, 270], [271, 275], [276, 278], [279, 282], [283, 287], [288, 296], [297, 299],
    # [300, 303], [304, 312], [313, 315], [316, 319], [320, 326], [327, 332], [332, 333], [334, 345], [346, 352],
    # [353, 356], [357, 365], [366, 368], [369, 372], [373, 379], [379, 380], [381, 382], [383, 389], [390, 395],
    # [396, 398], [399, 405], [406, 409], [410, 420], [420, 421], [422, 424], [425, 427], [428, 429], [430, 437],
    # [438, 440], [441, 444], [445, 451], [452, 454], [455, 462], [462, 463], [464, 470], [471, 476], [477, 480],
    # [481, 487], [488, 492], [493, 502], [503, 511], [512, 514], [515, 520], [521, 531], [532, 541], [542, 544],
    # [545, 549], [549, 550], [551, 553], [554, 557], [558, 561], [562, 564], [565, 568], [569, 573], [574, 579],
    # [580, 581], [581, 584], [585, 587], [588, 589], [590, 596], [597, 601], [602, 606], [607, 615], [616, 623],
    # [624, 625], [626, 633], [634, 637], [638, 641], [642, 646], [647, 651], [651, 652], [652, 653], [654, 656],
    # [657, 658], [659, 665], [665, 666], [667, 673], [674, 679], [680, 686], [687, 689], [690, 694], [694, 695]],
    # 'answers': ['Saint Bernadette Soubirous'], 'uuid': '5733be284776f41900661182'} 一共87599 个
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    # dev_eval_file与train_eval_file 格式一样
    # 一共10570 个
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    # meta 10482
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]
    print("BIDAF Building model...")
    parser = get_record_parser(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        # context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id是parse返回的数据具体的shape和数据类型为
        # <BatchDataset shapes: ((?, 400), (?, 50), (?, 400, 16), (?, 50, 16), (?, 400), (?, 400), (?,)),
        # types: (tf.int32, tf.int32, tf.int32, tf.int32, tf.float32, tf.float32, tf.int64)>
        train_data_set = get_batch_dataset(config.train_record_file, parser, config)
        dev_data_set = get_dataset(config.dev_record_file, parser, config)
        # feed able iterator https://www.bilibili.com/read/cv647026
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_data_set.output_types, train_data_set.output_shapes)
        train_iterator = train_data_set.make_one_shot_iterator()
        dev_iterator = dev_data_set.make_one_shot_iterator()

        model = Model(config, iterator, word_mat, char_mat, graph=g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        patience = 0
        best_f1 = 0.
        best_em = 0.

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.log_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            global_step = max(sess.run(model.global_step))

            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle,
                                                                                   model.dropout: config.dropout})
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    _, summ = evaluate_batch(
                        model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                    for s in summ:
                        writer.add_summary(s, global_step)

                    metrics, summ = evaluate_batch(
                        model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                    dev_f1 = metrics["f1"]
                    dev_em = metrics["exact_match"]
                    if dev_f1 < best_f1 and dev_em < best_em:
                        patience +=1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_em = max(best_em, dev_em)
                        best_f1 = max(best_f1, dev_f1)

                    for s in summ:
                        writer.add_summary(s, global_step)
                    writer.flush()
                    filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2 = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/f1".format(data_type), simple_value=metrics["f1"], )])
    em_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/em".format(data_type), simple_value=metrics["exact_match"])])
    return metrics, [loss_sum, f1_sum, em_sum]










