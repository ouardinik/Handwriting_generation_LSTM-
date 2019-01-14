import numpy as np
import scipy.stats
import tensorflow as tf

strokes = np.load('../data/strokes.npy')
stroke = strokes[0]


def generate_unconditionally(random_seed=1):
    #change desired length here
    desired_seq_length = 600
    n_units = 900
    batch_size = 1
    #redefine RNN the cell
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_units,state_is_tuple=True)
    init_sample = cell.zero_state(batch_size, dtype=tf.float32)
    #restore session
    sess = tf.Session()    
    # load meta graph and restore weights
    saver = tf.train.import_meta_graph('/home/khalilouardini/Desktop/lyrebird-egg-master/LSTM_900_2xclip.chkp.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/home/khalilouardini/Desktop/lyrebird-egg-master'))
    graph = tf.get_default_graph()
    #tf.saver.restore(sess, '/home/khalil/lyrebird-egg-master/HG_LSTM_900_2xclip.chkp')
    ######################### restore ops ##############################################
    sample_stroke = graph.get_tensor_by_name("op_to_restore:0")
    out_sample = graph.get_tensor_by_name("op_to_restore_1:0")
    sk_pi = graph.get_tensor_by_name("op_to_restore_2:0")
    sk_mu_x = graph.get_tensor_by_name("op_to_restore_3:0")
    sk_mu_y = graph.get_tensor_by_name("op_to_restore_4:0")
    sk_std_x = graph.get_tensor_by_name("op_to_restore_5:0")
    sk_std_y = graph.get_tensor_by_name("op_to_restore_6:0")
    sk_rho = graph.get_tensor_by_name("op_to_restore_7:0")
    sk_param_e = graph.get_tensor_by_name("op_to_restore_8:0")
    # initialization
    sample = np.zeros((desired_seq_length,3))
    current_stroke = np.zeros((1,1,3))
    previous_stroke = sess.run(cell.zero_state(batch_size,dtype=tf.float32))
    for k in range(desired_seq_length):
        feed_dict = {sample_stroke:current_stroke, init_sample:previous_stroke}
        
        [pi0, mu1, mu2, std1, std2, rho0, param_e0,next_stroke] = sess.run([sk_pi, sk_mu_x, sk_mu_y,
                                                                            sk_std_x, sk_std_y,
                                                                            sk_rho, sk_param_e, out_sample],feed_dict)
                                                                            
        #pick a random mixture
        mix = np.random.randint(0,20)
        mu = np.array([mu1[0,mix],mu2[0,mix]])
        sigma = np.array([[std1[0,mix]*std1[0,mix],rho0[0,mix]*std1[0,mix]*std2[0,mix]],
                          [rho0[0,mix]*std1[0,mix]*std2[0,mix],std2[0,mix]*std2[0,mix]]])
        #generate data
        z = np.random.multivariate_normal(mu, sigma, 1)
        z_e = scipy.stats.bernoulli.rvs(param_e0, size=1)
        sample[k] = [z_e, z[0,0], z[0,1]]

        current_stroke = np.zeros((1,1,3))
        current_stroke[0][0] = [z_e, z[0,0], z[0,1]]
        previous_stroke = next_stroke
    
    # Output:
    return sample

    #   sample - numpy 2D-array (T x 3)

def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'