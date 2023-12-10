import numpy as np
import warnings
from algorithm.utils import set_noisy_label
from algorithm.model import *
from algorithm.data_import import *
from absl import app, flags
from algorithm.model import CL_Fair_model


warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", "adult", "dataset name")
flags.DEFINE_float("c", 0.4, "bias ratio")
flags.DEFINE_integer("hidden_dim", 10, "hidden dimension")
flags.DEFINE_integer("out_dim", 2, "output dimension")
flags.DEFINE_integer("epochs", 50, "# of epochs")
flags.DEFINE_float("lr_1", 1e-2, "leraning rate for model a")
flags.DEFINE_float("lr_2", 5e-2, "leraning rate for model b")
flags.DEFINE_float("ns", 0.85, "n_s")
flags.DEFINE_float("sigma", 6e-1, "sigma")
flags.DEFINE_string("crop_type","asy","crop type")

def main(argv):
    input_data = pickle.load(open("dataset/{}/train.pkl".format(FLAGS.dataset),"rb"))
    target_data = pickle.load(open("dataset/{}/test.pkl".format(FLAGS.dataset),"rb"))


    N = input_data['x'].shape[0] + target_data['x'].shape[0]
    SA = len(np.where(input_data['s']==0)[0]) + len(np.where(target_data['s']==0)[0])
    SB = len(np.where(input_data['s']==1)[0]) + len(np.where(target_data['s']==1)[0])
    print("Dataset: {}\n#Total: {}, #Group A: {}, #Group B:{} ".format(FLAGS.dataset, N, SA, SB))

    data, clean_idx = set_noisy_label(input_data,crop_type=FLAGS.crop_type, crop_ratio=FLAGS.c)
    input_dim = data['x'].shape[1]
    print("Corrupt ratio: {}".format(FLAGS.c))

    train_loader, test_loader = load_data(data,target_data)

    # Collect results
    acc_list = []
    deo_list = []
    di_list = []
    dp_list = []
    # Model training
    print("Start training...")
    for i in range(10):
        print("-----Round: {}-----".format(i))
       
        Debiased_model = CL_Fair_model(input_dim,
                                       epoch = FLAGS.epochs,
                                       hidden_dim = FLAGS.hidden_dim, 
                                       output_dim = FLAGS.out_dim, 
                                       lr_1 = FLAGS.lr_1, 
                                       lr_2 = FLAGS.lr_2,
                                       n_s = FLAGS.ns,
                                       sigma = FLAGS.sigma)
        Debiased_model.train_coteaching(train_loader)
        r_dict = Debiased_model.test(test_loader)


        acc_list.append(r_dict['err'])
        deo_list.append(r_dict['deo'])
        di_list.append(r_dict['di'])
        dp_list.append(r_dict['dp'])

    print("ERR: {:.2f}+/-{:.2f}".format(np.mean(np.array(acc_list)), np.std(np.array(acc_list))))
    print("DEO: {:.2f}+/-{:.2f}".format(np.mean(np.array(deo_list)), np.std(np.array(deo_list))))
    print("DI: {:.2f}+/-{:.2f}".format(np.mean(np.array(di_list)), np.std(np.array(di_list))))
    print("DP: {:.2f}+/-{:.2f}".format(np.mean(np.array(dp_list)), np.std(np.array(dp_list))))



if __name__ == "__main__":
    app.run(main)
    
