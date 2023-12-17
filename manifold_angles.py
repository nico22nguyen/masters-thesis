import tensorflow as tf

class LayerCharacteristics():
  def __init__(self):
    self.space_linearity_sum = []
    self.total_sum = []
    self.svd_weights = {}
    self.distances = {}
    self.neighbour_dict = {}
    self.U_dict = {}

def ManifoldAngles(layerfeatlist, neighboursize1=10, classsize=10, dim_reduc_size=3):
  starttime = tf.timestamp()
  tf.print("start: ")

  no_of_layers = len(layerfeatlist)
  reduction_quality = []
  class_chars = []

  for c1 in range(classsize):
    class_chars.append([])
    for layer_i in range(no_of_layers):
      class_chars[c1].append(LayerCharacteristics())
      layer_features = layerfeatlist[layer_i]

      layer_start = tf.timestamp()

      for i,x_i in enumerate(layer_features):

        class_chars[c1][layer_i].neighbour_dict[i] = tf.argsort(tf.norm( tf.math.subtract(layer_features,x_i) ,axis=1))[0:neighboursize1+1]

        W_i = tf.gather(layer_features,class_chars[c1][0].neighbour_dict[i])

        W_i = ( W_i - tf.math.reduce_mean(W_i,axis=0) )
        s, u, v = tf.linalg.svd( W_i )
        W_i_reduced = v[:,:dim_reduc_size]

        class_chars[c1][layer_i].svd_weights[i] = s[:dim_reduc_size]
        reduction_quality.append(  tf.reduce_sum( (s[:dim_reduc_size])/ tf.reduce_sum(s) ) )
        class_chars[c1][layer_i].U_dict[i] = W_i_reduced

      tf.print("--layer time: ", tf.timestamp() - layer_start)
      class_chars[c1][layer_i].space_linearity_sum = 0.0
      angle_start = tf.timestamp()

      manifold_neighbour_angle_sum=[]
      for i in range(len(class_chars[c1][layer_i].U_dict)):
        manifold_neighbour_angle_sum_temp=[]
        manifold_neighbour_angle_sum.append([])

        for j in class_chars[c1][0].neighbour_dict[i]:
          if i != j:
            teta =  tf.matmul(  tf.transpose(class_chars[c1][layer_i].U_dict[i]),  class_chars[c1][layer_i].U_dict[int(j)]   )
            weights =  tf.matmul(  tf.transpose( tf.expand_dims(class_chars[c1][layer_i].svd_weights[i],0)), tf.expand_dims(class_chars[c1][layer_i].svd_weights[int(j)],0)  )
            Q = teta*weights

            s, u, v = tf.linalg.svd( Q )

            tetaw = tf.reduce_sum(s)/tf.linalg.trace(weights)
            angles = tf.math.acos( tf.clip_by_value(tetaw,-1,1) )
            manifold_neighbour_angle_sum_temp.append( tf.math.sin(angles)  )

        manifold_neighbour_angle_sum[i].append(tf.reduce_mean(tf.convert_to_tensor(manifold_neighbour_angle_sum_temp)))
      class_chars[c1][layer_i].space_linearity_sum = tf.reduce_mean( tf.convert_to_tensor(manifold_neighbour_angle_sum ))
      tf.print("--angle time: ", tf.timestamp() - angle_start)

  if no_of_layers==1: tf.print("Average reduction quality: ",  tf.reduce_mean(reduction_quality))
  tf.print("endtime: ", tf.timestamp() - starttime)
  return class_chars,manifold_neighbour_angle_sum