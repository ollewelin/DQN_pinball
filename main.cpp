
#include <iostream>
#include <stdio.h>
#include "fc_m_resnet.hpp"
#include "convolution.hpp"
#include <vector>
#include <time.h>
using namespace std;
#include <cstdlib>
#include <ctime>
#include <math.h>   // exp
#include <stdlib.h> // exit(0);
#include <cstdlib>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> // Basic OpenCV structures (cv::Mat, Scalar)
#include <iomanip> // for std::setprecision

#include "pinball_game.hpp"
#define MOVE_DOWN 0
#define MOVE_UP 1
#define MOVE_STOP 2

int main()
{
    cout << "Convolution neural network under work..." << endl;
    // ======== create 2 convolution layer objects =========
    convolution conv_L1;
    convolution conv_L2;
    //======================================================
    pinball_game gameObj1;///Instaniate the pinball game
    gameObj1.init_game();///Initialize the pinball game with serten parametrers
    gameObj1.slow_motion=0;///0=full speed game. 1= slow down
    gameObj1.replay_times = 0;///If =0 no replay. >0 this is the nuber of replay with serveral diffrent actions so the ageint take the best rewards before make any weights update
    gameObj1.advanced_game = 1;///0= only a ball. 1= ball give awards. square gives punish

    const int pixel_height = 50;///The input data pixel height, note game_Width = 220
    const int pixel_width = 50;///The input data pixel width, note game_Height = 200
    Mat resized_grapics, test, pix2hid_weight, hid2out_weight;
    Size size(pixel_width,pixel_height);//the dst image size,e.g.50x50
    const int nr_frames_strobed = 4;
    //=========== Neural Network size settings ==============
    fc_m_resnet fc_nn_end_block;
    string weight_filename_end;
    weight_filename_end = "end_block_weights.dat";
    string L1_kernel_k_weight_filename;
    L1_kernel_k_weight_filename = "L1_kernel_k.dat";
    string L1_kernel_b_weight_filename;
    L1_kernel_b_weight_filename = "L1_kernel_b.dat";
    string L2_kernel_k_weight_filename;
    L2_kernel_k_weight_filename = "L2_kernel_k.dat";
    string L2_kernel_b_weight_filename;
    L2_kernel_b_weight_filename = "L2_kernel_b.dat";
    fc_nn_end_block.get_version();
    fc_nn_end_block.block_type = 2;
    fc_nn_end_block.use_softmax = 1;
    fc_nn_end_block.activation_function_mode = 2;
    fc_nn_end_block.use_skip_connect_mode = 0; // 1 for residual network architetcture
    fc_nn_end_block.use_dropouts = 1;
    fc_nn_end_block.dropout_proportion = 0.4;
    conv_L1.get_version();

    //==== Set up convolution layers ===========
    cout << "conv_L1 setup:" << endl;
    
    int input_channels = 1;     //=== one channel MNIST dataset is used ====
    conv_L1.set_kernel_size(5); // Odd number
    conv_L1.set_stride(2);
    conv_L1.set_in_tensor(pixel_width*pixel_height*nr_frames_strobed, input_channels); // data_size_one_sample_one_channel, input channels
    conv_L1.set_out_tensor(30);                                              // output channels
    conv_L1.output_tensor.size();
    conv_L1.top_conv = 1;
    //========= L1 convolution (vectors) all tensor size for convolution object is finnish =============

    //==== Set up convolution layers ===========
    cout << "conv_L2 setup:" << endl;
    conv_L2.set_kernel_size(5); // Odd number
    conv_L2.set_stride(2);
    conv_L2.set_in_tensor((conv_L1.output_tensor[0].size() * conv_L1.output_tensor[0].size()), conv_L1.output_tensor.size()); // data_size_one_sample_one_channel, input channels
    conv_L2.set_out_tensor(25);                                                                                               // output channels
    conv_L2.output_tensor.size();
    const int end_inp_nodes = (conv_L2.output_tensor[0].size() * conv_L2.output_tensor[0].size()) * conv_L2.output_tensor.size();
    cout << "end_inp_nodes = " << end_inp_nodes << endl;
    const int end_hid_layers = 2;
    const int end_hid_nodes_L1 = 200;
    const int end_hid_nodes_L2 = 50;
    const int end_out_nodes = 10;
    for (int i = 0; i < end_inp_nodes; i++)
    {
        fc_nn_end_block.input_layer.push_back(0.0);
        fc_nn_end_block.i_layer_delta.push_back(0.0);
    }

    for (int i = 0; i < end_out_nodes; i++)
    {
        fc_nn_end_block.output_layer.push_back(0.0);
        fc_nn_end_block.target_layer.push_back(0.0);
    }
    fc_nn_end_block.set_nr_of_hidden_layers(end_hid_layers);
    fc_nn_end_block.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L1);
    fc_nn_end_block.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L2);
    //  Note that set_nr_of_hidden_nodes_on_layer_nr() cal must be exactly same number as the set_nr_of_hidden_layers(end_hid_layers)
    //============ Neural Network Size setup is finnish ! ==================

    //=== Now setup the hyper parameters of the Neural Network ====
    const double learning_rate_end = 0.001;
    fc_nn_end_block.momentum = 0.9;
    fc_nn_end_block.learning_rate = learning_rate_end;
    conv_L1.learning_rate = 0.0001;
    conv_L1.momentum = 0.9;
    conv_L2.learning_rate = 0.0001;
    conv_L2.momentum = 0.9;
    conv_L1.activation_function_mode = 2;
    conv_L2.activation_function_mode = 2;

    char answer;
    double init_random_weight_propotion = 0.25;
    double init_random_weight_propotion_conv = 0.25;
    cout << "Do you want to load kernel weights from saved weight file = Y/N " << endl;
    cin >> answer;
    if (answer == 'Y' || answer == 'y')
    {
        conv_L1.load_weights(L1_kernel_k_weight_filename, L1_kernel_b_weight_filename);
        conv_L2.load_weights(L2_kernel_k_weight_filename, L2_kernel_b_weight_filename);
        cout << "Do you want to randomize fully connected layers Y or N load weights  = Y/N " << endl;
        cin >> answer;
        if (answer == 'Y' || answer == 'y')
        {
            fc_nn_end_block.randomize_weights(init_random_weight_propotion);
        }
        else
        {
            fc_nn_end_block.load_weights(weight_filename_end);
        }
    }
    else
    {
        fc_nn_end_block.randomize_weights(init_random_weight_propotion);
        conv_L1.randomize_weights(init_random_weight_propotion_conv);
        conv_L2.randomize_weights(init_random_weight_propotion_conv);
    }
    //Set up a OpenCV mat
    srand(static_cast<unsigned>(time(NULL))); // Seed the randomizer


    //Save all weights
    fc_nn_end_block.save_weights(weight_filename_end);
    conv_L1.save_weights(L1_kernel_k_weight_filename, L1_kernel_b_weight_filename);
    conv_L2.save_weights(L2_kernel_k_weight_filename, L2_kernel_b_weight_filename);
    //End
}
