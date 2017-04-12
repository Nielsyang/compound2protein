def read_from_text(filename_queue, compound_len, protein_len):
  reader = tf.TextLineReader()
  _, value = reader.read(filename_queue)
  data = tf.decode_csv(value)
  example_c = tf.stack(data[:compound_len])
  example_p = tf.stack(data[compound_len:-1])
  label = tf.stack(data[-1])
  # processed_example = some_processing(example)
  return example_c, example_p, label

def batch_input(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example_c, example_p, label = read_my_file_format(filename_queue)
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch_c, example_batch_p, label_batch = tf.train.shuffle_batch(
      [example_c, example_p, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch_c, example_batch_p, label_batch

def model(compound, protein):
  img_c = generator_c(compound)
  img_p = generator_p(protein)
  feature_c = discriminator(img_c)
  feature_p = discriminator(img_p, reuse=True)
  out_c = discriminator_c(feature_c)
  out_p = discriminator_p(feature_p)
  out = tf.concat(3, [out_c, out_p])
  out = conv2d(out,2)
     
