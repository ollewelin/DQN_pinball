
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
#include <iomanip>               // for std::setprecision

#include "pinball_game.hpp"
#define MOVE_DOWN 0
#define MOVE_UP 1
#define MOVE_STOP 2


vector<int> fisher_yates_shuffle(vector<int> table);

int main()
{
    srand(static_cast<unsigned>(time(NULL))); // Seed the randomizer
    cout << "DQN program" << endl;
    int total_plays = 0;
    const int all_clip_der = 0;
    // ======== create convolution layer objects =========
    convolution conv_L1;

    //======================================================
    pinball_game gameObj1;      /// Instaniate the pinball game
    gameObj1.init_game();       /// Initialize the pinball game with serten parametrers
    gameObj1.slow_motion = 0;   /// 0=full speed game. 1= slow down
    gameObj1.replay_times = 0;  /// If =0 no replay. >0 this is the nuber of replay with serveral diffrent actions so the ageint take the best rewards before make any weights update
    gameObj1.advanced_game = 1; /// 0= only a ball. 1= ball give awards. square gives punish
    gameObj1.use_image_diff = 0;
    gameObj1.high_precition_mode = 1; /// This will make adjustable rewards highest at center of the pad.
    gameObj1.use_dice_action = 0;
    gameObj1.drop_out_percent = 0;
    gameObj1.Not_dropout = 1;
    gameObj1.flip_reward_sign = 0;
    gameObj1.print_out_nodes = 0;
    gameObj1.enable_ball_swan = 0;
    gameObj1.use_character = 0;
    gameObj1.enabel_3_state = 1; // Input Action from Agent. move_up: 1= Move up pad. 0= Move down pad. 2= STOP used only when enabel_3_state = 1

    // statistics report
    const int max_w_p_nr = 1000;
    int win_p_cnt = 0;
    int win_counter = 0;
    double last_win_probability = 0.5;
    double now_win_probability = last_win_probability;

    // Set up a OpenCV mat
    const int pixel_height = 45; /// The input data pixel height, note game_Width = 220
    const int pixel_width = 45;  /// The input data pixel width, note game_Height = 200
 
    // ======== create convolution layer objects =========
    const int nr_of_networks = 2;//0 = Policy network, 1 = Frozen Target network
    const int conv_layers = 3;
    const int see_nr_of_frames = 4;

    convolution conv_net[nr_of_networks][conv_layers][see_nr_of_frames];
    int conv_tensor_in_size[conv_layers];
    int conv_inp_channels[conv_layers];
    int conv_out_channels[conv_layers];
    int conv_kernel_size[conv_layers];
    int conv_stride[conv_layers];

    conv_tensor_in_size[0] = pixel_height * pixel_width;

    conv_out_channels[0] = 12;
    conv_out_channels[1] = 14;
    conv_out_channels[2] = 16;
    conv_kernel_size[0] = 5;
    conv_kernel_size[1] = 5;
    conv_kernel_size[2] = 3;
    conv_stride[0] = 2;
    conv_stride[1] = 2;
    conv_stride[2] = 1;

    for (int i = 0; i < nr_of_networks; i++)
    {
        for (int j = 0; j < conv_layers; j++)
        {
            for (int k = 0; k < see_nr_of_frames; k++)
            {
                conv_net[i][j][k].set_kernel_size(conv_kernel_size[j]); 
                conv_net[i][j][k].set_stride(conv_stride[j]); 
                if(j == 0)
                {
                    conv_net[i][j][k].set_in_tensor(conv_tensor_in_size[j], 1);
                }
                else
                {
                    conv_tensor_in_size[j] = (conv_net[i][j-1][k].output_tensor[0].size() * conv_net[i][j-1][k].output_tensor[0].size());
                    conv_net[i][j][k].set_in_tensor(conv_tensor_in_size[j], conv_out_channels[j-1]); 
                }
                conv_net[i][j][k].set_out_tensor(conv_out_channels[j]);
                conv_net[i][j][k].learning_rate = 0.001;
                conv_net[i][j][k].momentum = 0.0;//
                conv_net[i][j][k].activation_function_mode = 2;
            }
        }
    }
    // ========= Set up fully connected networks ============
    int end_inp_nodes = conv_net[0][conv_layers-1][0].output_tensor[0].size() * conv_net[0][conv_layers-1][0].output_tensor[0].size() * conv_net[0][conv_layers-1][0].output_tensor.size() * see_nr_of_frames;
    cout << "end_inp_nodes = " << end_inp_nodes << endl;
    const int end_hid_layers = 3;
    int end_hid_nodes_LX[end_hid_layers];
    end_hid_nodes_LX[0] = 200;
    end_hid_nodes_LX[1] = 80;
    end_hid_nodes_LX[2] = 40;
    const int end_out_nodes = 3; // Up, Down and Stop action

    int nr_of_fc_hid_layers = 3;
    fc_m_resnet fc_nn_end_block[nr_of_networks];

    



    for (int i = 0; i < nr_of_networks; i++)
    {
        fc_nn_end_block[i].get_version();
        fc_nn_end_block[i].block_type = 2;
        fc_nn_end_block[i].use_softmax = 0;                               // 0= Not softmax for DQN reinforcement learning
        fc_nn_end_block[i].activation_function_mode = 2;                  // ReLU for all fully connected activation functions except output last layer
        fc_nn_end_block[i].force_last_activation_function_to_sigmoid = 0; // 1 = Last output last layer will have Sigmoid functions regardless mode settings of activation_function_mode
        fc_nn_end_block[i].use_skip_connect_mode = 0;                     // 1 for residual network architetcture
        fc_nn_end_block[i].use_dropouts = 1;
        fc_nn_end_block[i].dropout_proportion = 0.5;
        fc_nn_end_block[i].clip_deriv = all_clip_der;

        fc_nn_end_block[i].momentum = 1.0;//Use 1.0 for fully connected when mini batch used for accumulated derivatives through the batch for optimizer
        fc_nn_end_block[i].learning_rate = 0.001;

        for (int j = 0; j < end_inp_nodes; j++)
        {
            fc_nn_end_block[i].input_layer.push_back(0.0);
            fc_nn_end_block[i].i_layer_delta.push_back(0.0);
        }

        for (int j = 0; j < end_out_nodes; j++)
        {
            fc_nn_end_block[i].output_layer.push_back(0.0);
            fc_nn_end_block[i].target_layer.push_back(0.0);
        }
        fc_nn_end_block[i].set_nr_of_hidden_layers(end_hid_layers);
        for (int j = 0; j < end_hid_layers; j++)
        {
            fc_nn_end_block[i].set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_LX[j]);
        }
        
    }

    int save_cnt = 0;
    const int save_after_nr = 1;
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
    string L3_kernel_k_weight_filename;
    L3_kernel_k_weight_filename = "L3_kernel_k.dat";
    string L3_kernel_b_weight_filename;
    L3_kernel_b_weight_filename = "L3_kernel_b.dat";

      
        
        
        
        
        // Save all weights
        if (save_cnt < save_after_nr)
        {
            save_cnt++;
        }
        else
        {
            save_cnt = 0;
//            fc_nn_end_block.save_weights(weight_filename_end);
            conv_L1.save_weights(L1_kernel_k_weight_filename, L1_kernel_b_weight_filename);
//            conv_L2.save_weights(L2_kernel_k_weight_filename, L2_kernel_b_weight_filename);
//            conv_L3.save_weights(L3_kernel_k_weight_filename, L3_kernel_b_weight_filename);
        }
}

vector<int> fisher_yates_shuffle(vector<int> table)
{
    int size = table.size();
    for (int i = 0; i < size; i++)
    {
        table[i] = i;
    }
    for (int i = size - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        int temp = table[i];
        table[i] = table[j];
        table[j] = temp;
    }
    return table;
}
