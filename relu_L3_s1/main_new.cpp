
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

//#define USE_MINIBATCH
#define Q_ALGORITHM_MODE_A

vector<int> fisher_yates_shuffle(vector<int> table);

int main()
{
    srand(static_cast<unsigned>(time(NULL))); // Seed the randomizer
    cout << "Convolution neural network under work..." << endl;
    int total_plays = 0;
    // ======== create 2 convolution layer objects =========
    convolution conv_L1;
    convolution conv_L2;
    convolution conv_L3;
    convolution conv_frozen_L1_target_net;
    convolution conv_frozen_L2_target_net;
    convolution conv_frozen_L3_target_net;

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



    // Set up a OpenCV mat
    const int pixel_height = 79; /// The input data pixel height, note game_Width = 220
    const int pixel_width = 79;  /// The input data pixel width, note game_Height = 200
    Mat resized_grapics, replay_grapics_buffert, game_video_full_size, upsampl_conv_view;
    Mat input_frm;

    Size image_size_reduced(pixel_height, pixel_width); // the dst image size,e.g.50x50

    vector<vector<int>> replay_actions_buffert;

    vector<vector<double>> rewards_at_game_replay;

    //=========== Neural Network size settings ==============
    fc_m_resnet fc_nn_end_block;
    fc_m_resnet fc_nn_frozen_target_net;
    int save_cnt = 0;
    const int save_after_nr = 0;
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

    const int all_clip_der = 0;

    fc_nn_end_block.get_version();
    fc_nn_end_block.block_type = 2;
    fc_nn_end_block.use_softmax = 0;                               // 0= Not softmax for DQN reinforcement learning
    fc_nn_end_block.activation_function_mode = 2;                  // ReLU for all fully connected activation functions except output last layer
    fc_nn_end_block.force_last_activation_function_to_sigmoid = 0; // 1 = Last output last layer will have Sigmoid functions regardless mode settings of activation_function_mode
    fc_nn_end_block.use_skip_connect_mode = 0;                     // 1 for residual network architetcture
    fc_nn_end_block.use_dropouts = 1;
    fc_nn_end_block.dropout_proportion = 0.5;
    fc_nn_end_block.clip_deriv = all_clip_der;


    fc_nn_frozen_target_net.block_type = fc_nn_end_block.block_type;
    fc_nn_frozen_target_net.use_softmax = fc_nn_end_block.use_softmax;
    fc_nn_frozen_target_net.force_last_activation_function_to_sigmoid = fc_nn_end_block.force_last_activation_function_to_sigmoid;
    fc_nn_frozen_target_net.use_skip_connect_mode = fc_nn_end_block.use_skip_connect_mode;
    fc_nn_frozen_target_net.use_dropouts = 0;
    fc_nn_frozen_target_net.clip_deriv = all_clip_der;

    conv_L1.get_version();

    //==== Set up convolution layers ===========
    cout << "conv_L1 setup:" << endl;
    const int nr_color_channels = 1;                 //=== 1 channel gray scale ====
    const int nr_frames_strobed = 7;                 // 4 Images in serie to make neural network to see movments
    const int L1_input_channels = nr_color_channels; // color channels
    const int L1_tensor_in_size = pixel_width * pixel_height;
    const int L1_tensor_out_channels = 10;
    const int L1_kernel_size = 7;
    const int L1_stride = 2;
    conv_L1.set_kernel_size(L1_kernel_size); // Odd number
    conv_L1.set_stride(L1_stride);
    conv_L1.set_in_tensor(L1_tensor_in_size, L1_input_channels); // data_size_one_sample_one_channel, input channels
    conv_L1.set_out_tensor(L1_tensor_out_channels);              // output channels
    conv_L1.top_conv = 0;
    conv_L1.clip_deriv = all_clip_der;

    // copy to a frozen copy network for target network
    conv_frozen_L1_target_net.set_kernel_size(L1_kernel_size);
    conv_frozen_L1_target_net.set_stride(L1_stride);
    conv_frozen_L1_target_net.set_in_tensor(L1_tensor_in_size, L1_input_channels);
    conv_frozen_L1_target_net.set_out_tensor(L1_tensor_out_channels);
    conv_frozen_L1_target_net.top_conv = conv_L1.top_conv;
    conv_frozen_L1_target_net.clip_deriv = conv_L1.clip_deriv;

    //==== Set up convolution layers ===========
    int L2_input_channels = conv_L1.output_tensor.size();
    int L2_tensor_in_size = (conv_L1.output_tensor[0].size() * conv_L1.output_tensor[0].size());
    int L2_tensor_out_channels = 12;
    int L2_kernel_size = 7;
    int L2_stride = 2;

    cout << "conv_L2 setup:" << endl;
    conv_L2.set_kernel_size(L2_kernel_size); // Odd number
    conv_L2.set_stride(L2_stride);
    conv_L2.set_in_tensor(L2_tensor_in_size, L2_input_channels); // data_size_one_sample_one_channel, input channels
    conv_L2.set_out_tensor(L2_tensor_out_channels);
    conv_L2.top_conv = 0;
    conv_L2.clip_deriv = all_clip_der;
    // copy to a frozen copy network for target network
    conv_frozen_L2_target_net.set_kernel_size(L2_kernel_size);
    conv_frozen_L2_target_net.set_stride(L2_stride);
    conv_frozen_L2_target_net.set_in_tensor(L2_tensor_in_size, L2_input_channels);
    conv_frozen_L2_target_net.set_out_tensor(L2_tensor_out_channels);
    conv_frozen_L2_target_net.top_conv = conv_L2.top_conv;
    conv_frozen_L2_target_net.clip_deriv = conv_L2.clip_deriv;

    //==== Set up convolution layers ===========
    int L3_input_channels = conv_L2.output_tensor.size();
    int L3_tensor_in_size = (conv_L2.output_tensor[0].size() * conv_L2.output_tensor[0].size());
    int L3_tensor_out_channels = 10;
    int L3_kernel_size = 5;
    int L3_stride = 1;

    cout << "conv_L3 setup:" << endl;
    conv_L3.set_kernel_size(L3_kernel_size); // Odd number
    conv_L3.set_stride(L3_stride);
    conv_L3.set_in_tensor(L3_tensor_in_size, L3_input_channels); // data_size_one_sample_one_channel, input channels
    conv_L3.set_out_tensor(L3_tensor_out_channels);
    conv_L3.top_conv = 0;
    conv_L3.clip_deriv = all_clip_der;
    // copy to a frozen copy network for target network
    conv_frozen_L3_target_net.set_kernel_size(L3_kernel_size);
    conv_frozen_L3_target_net.set_stride(L3_stride);
    conv_frozen_L3_target_net.set_in_tensor(L3_tensor_in_size, L3_input_channels);
    conv_frozen_L3_target_net.set_out_tensor(L3_tensor_out_channels);
    conv_frozen_L3_target_net.top_conv = conv_L3.top_conv;
    conv_frozen_L3_target_net.clip_deriv = conv_L3.clip_deriv;
    //========= L1,2,3 convolution (vectors) all tensor size for convolution object is finnish =============

    vector<vector<vector<vector<double>>>> conv_L1_stored_input_tensor;//4D [frames_stobed][input_channel][row][col].
    vector<vector<vector<vector<double>>>> conv_L2_stored_input_tensor;//4D [frames_stobed][input_channel][row][col].
    vector<vector<vector<vector<double>>>> conv_L3_stored_input_tensor;//4D [frames_stobed][input_channel][row][col].
    for(int i=0;i<nr_frames_strobed;i++)
    {
        conv_L1_stored_input_tensor.push_back(conv_L1.input_tensor);
        conv_L2_stored_input_tensor.push_back(conv_L2.input_tensor);
        conv_L3_stored_input_tensor.push_back(conv_L3.input_tensor);
    }
    // output channels
    int end_inp_nodes = (conv_L3.output_tensor[0].size() * conv_L3.output_tensor[0].size()) * conv_L3.output_tensor.size() * nr_frames_strobed;
    cout << "end_inp_nodes = " << end_inp_nodes << endl;
    const int end_hid_layers = 3;
    const int end_hid_nodes_L1 = 200;
    const int end_hid_nodes_L2 = 40;
    const int end_hid_nodes_L3 = 10;
    const int end_out_nodes = 3; // Up, Down and Stop action
    for (int i = 0; i < end_inp_nodes; i++)
    {
        fc_nn_end_block.input_layer.push_back(0.0);
        fc_nn_end_block.i_layer_delta.push_back(0.0);
        fc_nn_frozen_target_net.input_layer.push_back(0.0);
        fc_nn_frozen_target_net.i_layer_delta.push_back(0.0);
    }

    for (int i = 0; i < end_out_nodes; i++)
    {
        fc_nn_end_block.output_layer.push_back(0.0);
        fc_nn_end_block.target_layer.push_back(0.0);
        fc_nn_frozen_target_net.output_layer.push_back(0.0);
        fc_nn_frozen_target_net.target_layer.push_back(0.0);
    }
    fc_nn_end_block.set_nr_of_hidden_layers(end_hid_layers);
    fc_nn_end_block.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L1);
    fc_nn_end_block.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L2);
    fc_nn_end_block.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L3);
    fc_nn_frozen_target_net.set_nr_of_hidden_layers(end_hid_layers);
    fc_nn_frozen_target_net.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L1);
    fc_nn_frozen_target_net.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L2);
    fc_nn_frozen_target_net.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L3);
    //  Note that set_nr_of_hidden_nodes_on_layer_nr() cal must be exactly same number as the set_nr_of_hidden_layers(end_hid_layers)
    //============ Neural Network Size setup is finnish ! ==================

    //=== Now setup the hyper parameters of the Neural Network ====
    
    double target_off_level = 0.5; // OFF action target
    const double learning_rate_fc = 0.0001;
    const double learning_rate_conv = 0.0001;
    double learning_rate_end = learning_rate_fc;
    fc_nn_end_block.learning_rate = learning_rate_end;
    conv_L1.learning_rate = learning_rate_conv;
    conv_L2.learning_rate = learning_rate_conv;
    conv_L3.learning_rate = learning_rate_conv;
#ifdef USE_MINIBATCH
    fc_nn_end_block.momentum = 1.0;//1.0 for batch fc backpropagation
    conv_L1.momentum = 0.0;//0.0 for batch conv backpropagation
    conv_L2.momentum = 0.0;//0.0 for batch conv backpropagation
    conv_L3.momentum = 0.0;//0.0 for batch conv backpropagation
#else
    fc_nn_end_block.momentum = 0.9;//
    conv_L1.momentum = 0.9;//
    conv_L2.momentum = 0.9;//
    conv_L3.momentum = 0.9;//
#endif
    double init_random_weight_propotion = 0.55;
    double init_random_weight_propotion_conv = 0.35;
    const double warm_up_epsilon_start = 0.95;
    double warm_up_epsilon = warm_up_epsilon_start;
    const double warm_up_eps_derating = 0.15;
    const int warm_up_eps_nr = 3;
    int warm_up_eps_cnt = 0;
    const double start_epsilon = 0.60;
    const double stop_min_epsilon = 0.3;
  //  const int games_to_reach_stop_eps = 10000;
   // const double derating_epsilon = (stop_min_epsilon - start_epsilon) / (double)games_to_reach_stop_eps; // Derating speed per batch game
    const double derating_epsilon = 0.001;
    double dqn_epsilon = start_epsilon;   // Exploring vs exploiting parameter weight if dice above this threshold chouse random action. If dice below this threshold select strongest outoput action node
    if(warm_up_eps_nr > 0)
    {
        dqn_epsilon = warm_up_epsilon;
    }
    double gamma = 0.93f;
#ifndef Q_ALGORITHM_MODE_A
    double alpha = 0.8;
#endif
    const int g_replay_size = 2000;//Should be 10000 or more
    int update_frz_cnt = 0;
    // statistics report
    // const int max_w_p_nr = 1000;
    const int max_w_p_nr = g_replay_size;
    int win_p_cnt = 0;
    int win_counter = 0;
    double last_win_probability = 0.5;
    double now_win_probability = last_win_probability;

#ifdef USE_MINIBATCH   
    const int mini_batch_size = 32;
    int mini_batch_cnt = 0;
    const int update_frozen_after_samples = mini_batch_size * 8;
    //const int update_frozen_after_samples = 2000;
#else
    const int update_frozen_after_samples = 32 * 1;
#endif
    int debug_dec_act[g_replay_size];
    float debug_last_img_game_zero[L1_tensor_in_size];

    const int swapping_learning_mode = 0;
    const int swap_fc_conv_learn_after = 100;
    int swap_fc_conv_learn_cnt = 0;
    //==== Hyper parameter settings End ===========================

    //==== Set modes ===============
    conv_L1.activation_function_mode = 2;
    conv_L2.activation_function_mode = 2;
    conv_L3.activation_function_mode = 2;
    double visual_offset_conv = 0.5;
    if (conv_L3.activation_function_mode == 0)
    {
        visual_offset_conv = 0.0;
    }
    conv_frozen_L1_target_net.activation_function_mode = conv_L1.activation_function_mode;
    conv_frozen_L2_target_net.activation_function_mode = conv_L2.activation_function_mode;
    conv_frozen_L3_target_net.activation_function_mode = conv_L3.activation_function_mode;
    //==============================
    int grid_gap = 2;
    //  conv_out.create(conv_L2.output_tensor[0].size(),conv_L2.output_tensor.size() * grid_gap + conv_L2.output_tensor[0].size() * conv_L2.output_tensor[0][0].size(), CV_32FC1);
    int one_plane_L1_out_conv_size = conv_L1.output_tensor[0][0].size();
    cv::Mat Mat_L1_output_visualize(one_plane_L1_out_conv_size, conv_L1.output_tensor.size() * grid_gap + conv_L1.output_tensor.size() * one_plane_L1_out_conv_size, CV_32F); // Show a full pattern of L1 output convolution signals one rectangle for each output channel of L1 conv
    //
    int one_plane_L2_out_conv_size = conv_L2.output_tensor[0][0].size();
    cv::Mat Mat_L2_output_visualize(one_plane_L2_out_conv_size, conv_L2.output_tensor.size() * grid_gap + conv_L2.output_tensor.size() * one_plane_L2_out_conv_size, CV_32F); // Show a full pattern of L2 output convolution signals one rectangle for each output channel of L2 conv

    int one_plane_L3_out_conv_size = conv_L3.output_tensor[0][0].size();
    cv::Mat Mat_L3_output_visualize(one_plane_L3_out_conv_size, conv_L3.output_tensor.size() * grid_gap + conv_L3.output_tensor.size() * one_plane_L3_out_conv_size, CV_32F); // Show a full pattern of L3 output convolution signals one rectangle for each output channel of L3 conv

    // setup convolution kernels visualisation kernel_weights;//4D [output_channel][input_channel][kernel_row][kernel_col]

    cv::Mat visual_conv_kernel_L1_Mat((conv_L1.kernel_weights[0][0].size() + grid_gap) * conv_L1.kernel_weights[0].size(), (conv_L1.kernel_weights[0][0][0].size() + grid_gap) * conv_L1.output_tensor.size(), CV_32F);
    cv::Mat visual_conv_kernel_L2_Mat((conv_L2.kernel_weights[0][0].size() + grid_gap) * conv_L2.kernel_weights[0].size(), (conv_L2.kernel_weights[0][0][0].size() + grid_gap) * conv_L2.output_tensor.size(), CV_32F);
    cv::Mat visual_conv_kernel_L3_Mat((conv_L3.kernel_weights[0][0].size() + grid_gap) * conv_L3.kernel_weights[0].size(), (conv_L3.kernel_weights[0][0][0].size() + grid_gap) * conv_L3.output_tensor.size(), CV_32F);

    Mat upsampl_conv_view_2;
    
    int g_replay_nr = 0; // Used during play
    vector<int> g_replay_state_rand_list;
        int check_g_replay_nr = 0;                                                     // Used during replay training
 

    int single_game_state_size = gameObj1.nr_of_frames - nr_frames_strobed + 1; // the first for frames will not have any state
//    int check_state_nr = 0;                                                     // Used during replay training
    for (int i = 0; i < g_replay_size; i++)
    {
        for (int j = 0; j < single_game_state_size; j++)
        {
            g_replay_state_rand_list.push_back(0);
        }
    }

    char answer;
    cout << "Do you want to load kernel weights from saved weight file = Y/N " << endl;
    cin >> answer;
    if (answer == 'Y' || answer == 'y')
    {
        conv_L1.load_weights(L1_kernel_k_weight_filename, L1_kernel_b_weight_filename);
        conv_L2.load_weights(L2_kernel_k_weight_filename, L2_kernel_b_weight_filename);
        conv_L3.load_weights(L3_kernel_k_weight_filename, L3_kernel_b_weight_filename);
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
        conv_L3.randomize_weights(init_random_weight_propotion_conv);
    }

    cout << "gameObj1.gameObj1.game_Height " << gameObj1.game_Height << endl;
    cout << "gameObj1.gameObj1.game_Width " << gameObj1.game_Width << endl;

    // Start onlu one game now, only for get out size of grapichs to prepare memory
    gameObj1.replay_episode = 0;
    gameObj1.start_episode();

    game_video_full_size = gameObj1.gameGrapics.clone();
    resize(game_video_full_size, resized_grapics, image_size_reduced);
    imshow("resized_grapics", resized_grapics); ///  resize(src, dst, size);
    replay_grapics_buffert.create(pixel_height * gameObj1.nr_of_frames, pixel_width * g_replay_size, CV_32FC1);
    upsampl_conv_view.create(pixel_height, pixel_width, CV_32FC1);
    input_frm.create(pixel_height, pixel_width, CV_32FC1);

    cout << "replay_grapics_buffert rows = " << replay_grapics_buffert.rows << endl;
    cout << "replay_grapics_buffert cols = " << replay_grapics_buffert.cols << endl;
    cout << "resized_grapics rows = " << resized_grapics.rows << endl;
    cout << "resized_grapics cols = " << resized_grapics.cols << endl;

    vector<int> dummy_1D_vect_int;
    vector<double> dummy_1D_vect_double;
    for (int i = 0; i < g_replay_size; i++)
    {
        dummy_1D_vect_int.push_back(0); // Prepare an inner 1D vector to put to next replay_action_buffert 2D vector with size of g_replay_size
        dummy_1D_vect_double.push_back(0.0);
    }
    for (int i = 0; i < gameObj1.nr_of_frames; i++)
    {
        replay_actions_buffert.push_back(dummy_1D_vect_int); // Create the size of replay_action_buffert memory how store replay action of all states and the whole batch
        rewards_at_game_replay.push_back(dummy_1D_vect_double);    //
    }
    Mat debug;
    const int max_nr_epochs = 1000000;
    for (int epoch = 0; epoch < max_nr_epochs; epoch++)
    {
        cout << "******** Epoch number = " << epoch << " **********" << endl;
        cout << "dqn_epsilon = " << dqn_epsilon << endl;
        for (int g_replay_cnt = 0; g_replay_cnt < g_replay_size; g_replay_cnt++)
        {

            g_replay_nr = g_replay_cnt;
            gameObj1.start_episode();
      //      cout << "Run one game and store it in replay memory index at g_replay_cnt = " << g_replay_cnt << endl;

            for (int frame_g = 0; frame_g < gameObj1.nr_of_frames; frame_g++) // Loop throue each of the 100 frames
            {
                gameObj1.frame = frame_g;
                gameObj1.run_episode();
                game_video_full_size = gameObj1.gameGrapics.clone();
                resize(game_video_full_size, resized_grapics, image_size_reduced);
                imshow("resized_grapics", resized_grapics); //  resize(src, dst, size);
                // Insert resized_grapics into replay_grapics_buffert below
                // replay_grapics_buffert filled with data from resized_grapics rows from 0 to pixel_height-1 into replay_grapics_buffert rows index pixel_height * frame_g + (resized_grapics rows from 0 to pixel_height-1)
                // Insert resized_grapics into replay_grapics_buffert
                // Calculate the starting column index for the ROI in replay_grapics_buffert
                int startCol = pixel_width * g_replay_nr;
                // Create a Rect to define the ROI in replay_grapics_buffert
                cv::Rect roi(startCol, pixel_height * frame_g, pixel_width, pixel_height);
                // Copy the relevant region from resized_grapics to replay_grapics_buffert
                resized_grapics(cv::Rect(0, 0, pixel_width, pixel_height)).copyTo(replay_grapics_buffert(roi));
                //  replay_grapics_buffert(roi).copyTo(debug);
                //  imshow("debug", debug);
                //  waitKey(1);
                if (frame_g >= nr_frames_strobed - 1) // Wait until all 4 images is up there in the game after start
                {
                    // Put in data from replay_grapics_buffert to L1_tensor_in_size
                    for (int f = 0; f < nr_frames_strobed; f++)
                    {

                        for (int i = 0; i < L1_input_channels; i++)
                        {
                            for (int j = 0; j < L1_tensor_in_size; j++)
                            {
                                int row = j / pixel_width;
                                int col = j % pixel_width;
                                float pixelValue = replay_grapics_buffert.at<float>(pixel_height * (frame_g - (nr_frames_strobed - 1) + f) + row, col + pixel_width * g_replay_nr);
                                conv_L1.input_tensor[i][row][col] = pixelValue;

                                if (g_replay_cnt == 0 && frame_g == gameObj1.nr_of_frames - 2)
                                {
                                    debug_last_img_game_zero[j] = pixelValue;
                                }
                            }
                        }

                        if (g_replay_cnt == g_replay_nr - 1)
                        {
                            // Debug
                            for (int i = 0; i < L1_input_channels; i++)
                            {
                                for (int j = 0; j < pixel_height; j++)
                                {
                                    for (int k = 0; k < pixel_width; k++)
                                    {
                                        float inp_frm = conv_L1.input_tensor[i][j][k];
                                        //      cout << "inp = " << inp_frm << endl;
                                        input_frm.at<float>(pixel_height * i + j, k) = inp_frm;
                                    }
                                }
                            }
                            imshow("input_frm", input_frm);
                            waitKey(1);
                            // end Debug
                        }

                        //**********************************************************************
                        //****************** Forward Pass training network *********************
                        conv_L1.conv_forward1();
                        conv_L2.input_tensor = conv_L1.output_tensor;
                        conv_L2.conv_forward1();
                        conv_L3.input_tensor = conv_L2.output_tensor;
                        conv_L3.conv_forward1();

                        int L3_out_one_side = conv_L3.output_tensor[0].size();
                        int L3_out_ch = conv_L3.output_tensor.size();
                        for (int oc = 0; oc < L3_out_ch; oc++)
                        {
                            for (int yi = 0; yi < L3_out_one_side; yi++)
                            {
                                for (int xi = 0; xi < L3_out_one_side; xi++)
                                {
                                    /*
                                     double out_conv = conv_L3.output_tensor[oc][yi][xi];
                                       if(yi== L3_out_one_side-1)
                                    {
                                        out_conv = 0.1;
                                    }

                            */
                                    fc_nn_end_block.input_layer[f * oc * L3_out_one_side * L3_out_one_side + oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi] = conv_L3.output_tensor[oc][yi][xi];
                                    //           cout << "conv_L3.output_tensor[" << oc << "][" << yi << "["  << xi << "] = "  << conv_L3.output_tensor[oc][yi][xi] << endl;
                                }
                            }
                        }
                    }

                    if (g_replay_cnt == 0)
                    {

                        // Visualization of L1 conv output
                        for (int oc = 0; oc < (int)conv_L1.output_tensor.size(); oc++)
                        {
                            for (int yi = 0; yi < one_plane_L1_out_conv_size; yi++)
                            {
                                for (int xi = 0; xi < one_plane_L1_out_conv_size; xi++)
                                {
                                    int visual_col = xi + (oc * grid_gap + oc * one_plane_L1_out_conv_size);
                                    int visual_row = yi;
                                    double pixel_data = conv_L1.output_tensor[oc][yi][xi];
                                    Mat_L1_output_visualize.at<float>(visual_row, visual_col) = (float)pixel_data + visual_offset_conv;
                                    //          cout <<"L1 out pixel = " << pixel_data << endl;
                                }
                            }
                        }

                        cv::imshow("Convolution L1 output", Mat_L1_output_visualize);
                        waitKey(1);
                        // Visualization of L2 conv output

                        for (int oc = 0; oc < (int)conv_L2.output_tensor.size(); oc++)
                        {
                            for (int yi = 0; yi < one_plane_L2_out_conv_size; yi++)
                            {
                                for (int xi = 0; xi < one_plane_L2_out_conv_size; xi++)
                                {
                                    int visual_col = xi + (oc * grid_gap + oc * one_plane_L2_out_conv_size);
                                    int visual_row = yi;
                                    double pixel_data = conv_L2.output_tensor[oc][yi][xi];
                                    Mat_L2_output_visualize.at<float>(visual_row, visual_col) = (float)pixel_data + visual_offset_conv;
                                    //        cout <<"L2 out pixel = " << pixel_data << endl;
                                }
                            }
                        }

                        cv::imshow("Convolution L2 output", Mat_L2_output_visualize);
                        waitKey(1);
                        // Visualization of L3 conv output

                        for (int oc = 0; oc < (int)conv_L3.output_tensor.size(); oc++)
                        {
                            for (int yi = 0; yi < one_plane_L3_out_conv_size; yi++)
                            {
                                for (int xi = 0; xi < one_plane_L3_out_conv_size; xi++)
                                {
                                    int visual_col = xi + (oc * grid_gap + oc * one_plane_L3_out_conv_size);
                                    int visual_row = yi;
                                    double pixel_data = conv_L3.output_tensor[oc][yi][xi];
                                    Mat_L3_output_visualize.at<float>(visual_row, visual_col) = (float)pixel_data + visual_offset_conv;
                                    //        cout <<"L3 out pixel = " << pixel_data << endl;
                                }
                            }
                        }

                        cv::imshow("Convolution L3 output", Mat_L3_output_visualize);
                        waitKey(1);
                    }

                    // Start Forward pass fully connected network
                    fc_nn_end_block.forward_pass(); // Forward pass though fully connected network
                    float exploring_dice = (float)(rand() % 65535) / 65536; // Through a fair dice. Random value 0..1.0 range
                    int decided_action = 0;
                    double max_decision = 0.0f;
                    if (exploring_dice < dqn_epsilon)
                    {
                        // Choose dice action (Exploration mode)
                        for (int i = 0; i < end_out_nodes; i++) // end_out_nodes = numbere of actions
                        {
                            float action_dice = (float)(rand() % 65535) / 65536; // Through a fair dice. Random value 0..1.0 range
                            // cout << "action_dice = " << action_dice << endl;
                            if (action_dice > (float)max_decision)
                            {
                                max_decision = (float)action_dice;
                                decided_action = i;
                            }
                        }
                    }
                    else
                    {
                        // Choose predicted action (Exploit mode)
                        for (int i = 0; i < end_out_nodes; i++)
                        {
                            // end_out_nodes = numbere of actions
                            double action_node = fc_nn_end_block.output_layer[i];
                            if (action_node > max_decision)
                            {
                                max_decision = action_node;
                                decided_action = i;
                            }
                            if (g_replay_cnt == 0)
                            {
                                cout << "action_node = " << action_node << " i = " << i << endl;
                            }
                        }
                    }
                    //  std::cout << std::fixed << std::setprecision(20);
                    //  cout << "decided_action = " << decided_action << endl;
                    //  cout << "exploring_dice = " << exploring_dice << endl;

                    gameObj1.move_up = decided_action; // Input Action from Agent. 1= Move up pad. 0= Move down pad. 2= STOP used only when enabel_3_state = 1
                //    cout << " decided_action = " << decided_action << endl;
                    replay_actions_buffert[frame_g][g_replay_nr] = decided_action;
 
                    //****************** Forward Pass training network complete ************
                    //**********************************************************************
                }
            }
            double abs_diff = abs(gameObj1.pad_ball_diff);
            
           // cout << "pad_ball_diff = " << abs_diff << endl;
            double rewards = 0.0;
            if(abs_diff< 1.0)
            {
                abs_diff = 1.0;
            }

            if (gameObj1.win_this_game == 1)
            {
                if(gameObj1.square == 1)
                {
                    
                    rewards = 5.0; // Win Rewards avoid square
             //       rewards /= abs_diff;
                }
                else
                {
                    rewards = 10.0; // Win Rewards catch ball
             //       rewards /= abs_diff;
                }
                win_counter++;
            }
            else
            {
                if(gameObj1.square == 1)
                {
                  //  rewards = -2.35; // Lose Penalty
                  rewards = -3.0;
                    //rewards /= abs_diff;
                }
                else
                {
                   // rewards = -3.95; // Lose Penalty
                   rewards = -3.0;
                    //rewards *= abs_diff;
                }
            }
            /*
            if(gameObj1.square == 1)
            {
                cout << "Game through a rectangle " << endl;
            }
            else
            {
                cout << "Game through a ball " << endl;
            }
            */
            // cout << " Rewards = " << rewards;


            cout << "                                                                                                       " << endl;
            std::cout << "\033[F";
            cout << "Play g_replay_cnt = " << g_replay_cnt << " Rewards = " << rewards << endl;
            // Move the cursor up one line (ANSI escape code)
            std::cout << "\033[F";

            rewards_at_game_replay[gameObj1.nr_of_frames - 1][g_replay_nr] = rewards;
            total_plays++;

//                const int swapping_learning_mode = 1;
//    const int swap_fc_conv_learn_after = 1000;
//    int swap_fc_conv_learn_cnt = 0;
            if (swapping_learning_mode == 1)
            {
                if (swap_fc_conv_learn_cnt < swap_fc_conv_learn_after)
                {
                    swap_fc_conv_learn_cnt++;
                }
                else
                {
                    swap_fc_conv_learn_cnt = 0;
                }
                if (swap_fc_conv_learn_cnt < swap_fc_conv_learn_after/2)
                {
                    fc_nn_end_block.learning_rate = learning_rate_fc;
                    conv_L1.learning_rate = 0.0;
                    conv_L2.learning_rate = 0.0;
                    conv_L3.learning_rate = 0.0;
                }
                else
                {
                    fc_nn_end_block.learning_rate = 0.0;
                    conv_L1.learning_rate = learning_rate_conv;
                    conv_L2.learning_rate = learning_rate_conv;
                    conv_L3.learning_rate = learning_rate_conv;
                }
            }
            // Calculate win probablilty
            if (win_p_cnt > 10)
            {
                now_win_probability = (double)win_counter / (double)(win_p_cnt + 1);
                if (g_replay_nr == g_replay_size - 1)
                {
                    cout << "Win probaility Now = " << now_win_probability * 100.0 << "% at play count = " << win_p_cnt + 1 << " Old win probablilty = " << last_win_probability * 100.0 << "% total plays = " << total_plays << endl;
                }
            }
            else
            {
                now_win_probability = 0.5;
            }
            if (win_p_cnt < max_w_p_nr)
            {
                win_p_cnt++;
            }
            else
            {
                win_p_cnt = 0;
                win_counter = 0;
                // Store last 1000 win probablilty
                last_win_probability = now_win_probability;
            }

          

        }
  //      imshow("replay_grapics_buffert", replay_grapics_buffert);
  //      waitKey(1);
          
    if (dqn_epsilon > stop_min_epsilon)
    {
        if (warm_up_eps_cnt < warm_up_eps_nr)
        {
            dqn_epsilon -= warm_up_eps_derating;
            if (dqn_epsilon < start_epsilon)
            {
                dqn_epsilon = start_epsilon; // Limit warm up warm_up_eps_derating if go below the start_epsilon value during warm up epsilon
            }
            warm_up_eps_cnt++;
        }
        else
        {
             dqn_epsilon -= derating_epsilon;
        }
    }

        // visual_conv_kernel_L1_Mat
        int kernel_output_channels = conv_L1.kernel_weights.size();
        int kernel_input_channels = conv_L1.kernel_weights[0].size();
        int kernel_side = conv_L1.kernel_weights[0][0].size();
        for (int oc = 0; oc < kernel_output_channels; oc++)
        {
            for (int ic = 0; ic < kernel_input_channels; ic++)
            {
                for (int yi = 0; yi < kernel_side; yi++)
                {
                    for (int xi = 0; xi < kernel_side; xi++)
                    {
                        int visual_col = xi + (oc * (kernel_side + grid_gap));
                        int visual_row = yi + ic * (kernel_side + grid_gap);
                        double pixel_data = conv_L1.kernel_weights[oc][ic][yi][xi]; // 4D [output_channel][input_channel][kernel_row][kernel_col]
                        visual_conv_kernel_L1_Mat.at<float>(visual_row, visual_col) = (float)pixel_data + 0.5;
                    }
                }
            }
        }
        cv::imshow("Kernel L1 ", visual_conv_kernel_L1_Mat);
        waitKey(1);
        // visual_conv_kernel_L2_Mat
        kernel_output_channels = conv_L2.kernel_weights.size();
        kernel_input_channels = conv_L2.kernel_weights[0].size();
        kernel_side = conv_L2.kernel_weights[0][0].size();
        for (int oc = 0; oc < kernel_output_channels; oc++)
        {
            for (int ic = 0; ic < kernel_input_channels; ic++)
            {
                for (int yi = 0; yi < kernel_side; yi++)
                {
                    for (int xi = 0; xi < kernel_side; xi++)
                    {
                        int visual_col = xi + (oc * (kernel_side + grid_gap));
                        int visual_row = yi + ic * (kernel_side + grid_gap);
                        double pixel_data = conv_L2.kernel_weights[oc][ic][yi][xi]; // 4D [output_channel][input_channel][kernel_row][kernel_col]
                        visual_conv_kernel_L2_Mat.at<float>(visual_row, visual_col) = (float)pixel_data + 0.5;
                    }
                }
            }
        }
        cv::imshow("Kernel L2 ", visual_conv_kernel_L2_Mat);
        waitKey(1);
        // visual_conv_kernel_L3_Mat
        kernel_output_channels = conv_L3.kernel_weights.size();
        kernel_input_channels = conv_L3.kernel_weights[0].size();
        kernel_side = conv_L3.kernel_weights[0][0].size();
        for (int oc = 0; oc < kernel_output_channels; oc++)
        {
            for (int ic = 0; ic < kernel_input_channels; ic++)
            {
                for (int yi = 0; yi < kernel_side; yi++)
                {
                    for (int xi = 0; xi < kernel_side; xi++)
                    {
                        int visual_col = xi + (oc * (kernel_side + grid_gap));
                        int visual_row = yi + ic * (kernel_side + grid_gap);
                        double pixel_data = conv_L3.kernel_weights[oc][ic][yi][xi]; // 4D [output_channel][input_channel][kernel_row][kernel_col]
                        visual_conv_kernel_L3_Mat.at<float>(visual_row, visual_col) = (float)pixel_data + 0.5;
                    }
                }
            }
        }
        cv::imshow("Kernel L3 ", visual_conv_kernel_L3_Mat);
        waitKey(100);

        //******************** Go through the batch of replay memory *******************
        cout << "********************************************************************************" << endl;
        cout << "********* Run the whole replay batch memory and training the DQN network *******" << endl;
        cout << "********************************************************************************" << endl;
        //   cout << "single_game_state_size = " << single_game_state_size << endl;
        g_replay_state_rand_list = fisher_yates_shuffle(g_replay_state_rand_list);
        int replay_decided_action = 0;



        fc_nn_end_block.clear_batch_accum();
        conv_L3.clear_kernel_delta();
        conv_L2.clear_kernel_delta();
        conv_L1.clear_kernel_delta();
        conv_L3.clear_i_tens_delta();
        conv_L2.clear_i_tens_delta();
        conv_L1.clear_i_tens_delta();

        for (int g_replay_state_cnt = 0; g_replay_state_cnt < (single_game_state_size * g_replay_size); g_replay_state_cnt++)
        {
            check_g_replay_nr = g_replay_state_rand_list[g_replay_state_cnt];
            //     cout << "Run one training state sample at replay memory at check_state_nr = " << check_state_nr << endl;
            g_replay_nr = check_g_replay_nr / single_game_state_size;
            //    cout << "Run one training state sample at g_replay_nr = " << g_replay_nr << endl;
            int single_game_frame_state = check_g_replay_nr % single_game_state_size;
         //   int frame_state = single_game_frame_state;
            //    cout << "single_game_frame_state = " << single_game_frame_state << endl;
            double max_Q_target_value = 0.0;
            int terminal_state = 0;  

            int L3_out_one_side = conv_L3.output_tensor[0].size();
            int L3_out_ch = conv_L3.output_tensor.size();
            {
                if (single_game_frame_state < single_game_state_size)
                {
                    
                    // Calculate the starting column index for the ROI in replay_grapics_buffert
                    int startCol = pixel_width * g_replay_nr;
                    int startRow = pixel_height * single_game_frame_state;

                    cv::Rect replay_roi(startCol, startRow, pixel_width, pixel_height * nr_frames_strobed);
                    for (int f = 0; f < nr_frames_strobed; f++)
                    {
                        for (int i = 0; i < L1_input_channels; i++)
                        {
                            for (int j = 0; j < L1_tensor_in_size; j++)
                            {
                                int row = j / pixel_width;
                                int col = j % pixel_width;

                                // Calculate the actual row and column indices in replay_roi
                                int roi_row = row + replay_roi.y + f * pixel_height;
                                int roi_col = col + replay_roi.x;

                                // Ensure the indices are within bounds
                                if (roi_row < replay_roi.y + replay_roi.height && roi_col < replay_roi.x + replay_roi.width)
                                {
                                    // Extract replay_roi data here to pixelValue
                                    float pixelValue = replay_grapics_buffert.at<float>(roi_row, roi_col);
                                    conv_L1.input_tensor[i][row][col] = pixelValue;
                                    if (g_replay_state_cnt  == 0 && single_game_frame_state == gameObj1.nr_of_frames - 2)
                                    {
                                        debug_last_img_game_zero[i] = pixelValue;
                                    }
                                }
                                else
                                {
                                    // Handle the case where the indices are out of bounds
                                    // You might want to set a default value or handle it differently based on your requirements.
                                    cout << "error case where the indices are out of bounds" << endl;
                                    conv_L1.input_tensor[i][row][col] = 0.0; // Set a default value to 0.0 for example
                                }
                            }
                        }
                        //**********************************************************************
                        //****************** Forward Pass training network *********************
                        conv_L1.conv_forward1();
                        conv_L2.input_tensor = conv_L1.output_tensor;
                        conv_L2.conv_forward1();
                        conv_L3.input_tensor = conv_L2.output_tensor;
                        conv_L3.conv_forward1();
                        conv_L1_stored_input_tensor[f] = conv_L1.input_tensor;
                        conv_L2_stored_input_tensor[f] = conv_L2.input_tensor;
                        conv_L3_stored_input_tensor[f] = conv_L3.input_tensor;

                        for (int oc = 0; oc < L3_out_ch; oc++)
                        {
                            for (int yi = 0; yi < L3_out_one_side; yi++)
                            {
                                for (int xi = 0; xi < L3_out_one_side; xi++)
                                {

                                    //                             double out_conv = conv_L3.output_tensor[oc][yi][xi];
                                    //         if(yi== L3_out_one_side-1)
                                    //    {
                                    //       out_conv = 0.1;
                                    //  }
                                    // fc_nn_end_block.input_layer[oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi] =out_conv ;

                                    fc_nn_end_block.input_layer[f * oc * L3_out_one_side * L3_out_one_side + oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi] = conv_L3.output_tensor[oc][yi][xi];
                                }
                            }
                        }
                    }
                    // Start Forward pass fully connected network
                    fc_nn_end_block.forward_pass(); // Forward pass though fully connected network

                    //****************** Forward Pass training network complete ************
                    //**********************************************************************

                    // replay_decided_action = replay_actions_buffert[single_game_frame_state + nr_frames_strobed - 1][g_replay_nr];orginal bug
                    int act_idx = single_game_frame_state + nr_frames_strobed;
                    if (act_idx >= gameObj1.nr_of_frames)
                    {
                        cout << "Error act_idx = " << act_idx << " gameObj1.nr_of_frames = " << gameObj1.nr_of_frames << endl;
                    }
                    replay_decided_action = replay_actions_buffert[act_idx][g_replay_nr];
                    if (single_game_frame_state == single_game_state_size - 2)
                    {
                        if (debug_dec_act[g_replay_nr] != replay_decided_action)
                        {
                            cout << endl;
                            cout << " Error debug_dec_act[g_replay_nr] = " << debug_dec_act[g_replay_nr] << " replay_decided_action = " << replay_decided_action << endl;
                        }
                    }


            //       replay_decided_action = replay_actions_buffert[single_game_frame_state + nr_frames_strobed - 1][g_replay_nr]; bug - 1
            //        cout <<" replay_decided_action = " << replay_decided_action  << endl;


                    single_game_frame_state++; // Take NEXT state to peak into and get next state Q-value for a target value to train on
                    // Calculate the starting column index for the ROI in replay_grapics_buffert
                    startCol = pixel_width * g_replay_nr;
                    startRow = pixel_height * single_game_frame_state;
                    cv::Rect replay_roi_2(startCol, startRow, pixel_width, pixel_height * nr_frames_strobed);
                    for (int f = 0; f < nr_frames_strobed; f++)
                    {
                        for (int i = 0; i < L1_input_channels; i++)
                        {
                            for (int j = 0; j < L1_tensor_in_size; j++)
                            {
                                int row = j / pixel_width;
                                int col = j % pixel_width;

                                // Calculate the actual row and column indices in replay_roi
                                int roi_row = row + replay_roi_2.y + f * pixel_height;
                                int roi_col = col + replay_roi_2.x;

                                // Ensure the indices are within bounds
                                if (roi_row < replay_roi_2.y + replay_roi_2.height && roi_col < replay_roi_2.x + replay_roi_2.width)
                                {
                                    // Extract replay_roi data here to pixelValue
                                    float pixelValue = replay_grapics_buffert.at<float>(roi_row, roi_col);
                                    conv_frozen_L1_target_net.input_tensor[i][row][col] = pixelValue;

                                    if (g_replay_nr == 0 && single_game_frame_state == single_game_state_size - 2)
                                    {
                                        if (debug_last_img_game_zero[j] != pixelValue)
                                        {
                                            cout << " Error debug_last_img_game_zero[" << j << "] = " << debug_last_img_game_zero[j] << " but pixelValue = " << pixelValue << endl;
                                        }
                                    }
                                }
                                else
                                {
                                    // Handle the case where the indices are out of bounds
                                    // You might want to set a default value or handle it differently based on your requirements.
                                    cout << "error case where the indices are out of bounds" << endl;
                                    conv_frozen_L1_target_net.input_tensor[i][row][col] = 0.0; // Set a default value to 0.0 for example
                                }
                            }
                        }

                        //======================================================================
                        //================== Forward Pass Frozen network NEXT state ============
                        conv_frozen_L1_target_net.conv_forward1();
                        conv_frozen_L2_target_net.input_tensor = conv_frozen_L1_target_net.output_tensor;
                        conv_frozen_L2_target_net.conv_forward1();
                        conv_frozen_L3_target_net.input_tensor = conv_frozen_L2_target_net.output_tensor;
                        conv_frozen_L3_target_net.conv_forward1();
                        for (int oc = 0; oc < L3_out_ch; oc++)
                        {
                            for (int yi = 0; yi < L3_out_one_side; yi++)
                            {
                                for (int xi = 0; xi < L3_out_one_side; xi++)
                                {
                                    //                                          double out_conv = conv_frozen_L3_target_net.output_tensor[oc][yi][xi];
                                    //                if(yi== L3_out_one_side-1)
                                    //             {
                                    //                 out_conv = 0.1;
                                    //             }
                                    //    fc_nn_frozen_target_net.input_layer[oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi] =out_conv ;

                                    fc_nn_frozen_target_net.input_layer[f * oc * L3_out_one_side * L3_out_one_side + oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi] = conv_frozen_L3_target_net.output_tensor[oc][yi][xi];
                                }
                            }
                        }
                    }
                    // Start Forward pass fully connected network
                    fc_nn_frozen_target_net.forward_pass(); // Forward pass though fully connected network
                    //================== Forward Pass Frozen network complete ==============
                    //======================================================================


                    // Search for max Q-value
                    max_Q_target_value = 0.0;

                    for (int i = 0; i < end_out_nodes; i++)
                    {
                        double action_node = fc_nn_frozen_target_net.output_layer[i];
                        if (action_node > max_Q_target_value)
                        {
                            max_Q_target_value = action_node;
                        }
                    }
                    terminal_state = 0;
                }
                else
                {
                    // End game state
                    terminal_state = 1;
                    max_Q_target_value = target_off_level; // Zero Q value at end state Only rewards will be used
              //      cout << "Replay END State at g_replay_nr = " << g_replay_nr << endl;
                }
                int rewards_idx_state = single_game_frame_state + nr_frames_strobed - 1;
                double rewards_transition_to = rewards_at_game_replay[rewards_idx_state][g_replay_nr];
                
//                int rewards_idx_state = single_game_frame_state + nr_frames_strobed - 1;
                // cout << "rewards_idx_state = " << rewards_idx_state << endl;
//                double rewards_here = rewards_at_game_replay[rewards_idx_state][g_replay_nr];
                //     double target_value = rewards_here + gamma * max_Q_target_value;
                //        #Q table UPDATE
                //        Q[state,action] = Q[state,action] + ALPHA * (reward + GAMMA * np.max(Q[state_next,:]) - Q[state,action])
                //   double target_value = rewards_here + gamma * (max_Q_target_value - );
                // decided_action

                for (int i = 0; i < end_out_nodes; i++)
                {

                    if (replay_decided_action == i)
                    {
                        if (terminal_state == 0)
                        {
#ifdef Q_ALGORITHM_MODE_A
                            // target_value = rewards_here + gamma * (max_Q_target_value - );
                            fc_nn_end_block.target_layer[i] = rewards_transition_to + gamma * max_Q_target_value;
#else
                            // Q[state,action] = Q[state,action] + ALPHA * (reward + GAMMA * np.max(Q[state_next,:]) - Q[state,action])
                            fc_nn_end_block.target_layer[i] = fc_nn_end_block.target_layer[i] + alpha * (rewards_here + gamma * max_Q_target_value - fc_nn_end_block.target_layer[i]);
#endif
                        }
                        else
                        {
                            fc_nn_end_block.target_layer[i] = rewards_transition_to; // Terminal state then the rewards_transition_to is the rewards at this terminal state
                        }
                    }
                    else
                    {
                        // fc_nn_end_block.target_layer[i] = target_off_level;
                        fc_nn_end_block.target_layer[i] = fc_nn_end_block.target_layer[i]; // No change
                    }
                }

                //fc_nn_end_block.backpropagtion_and_update();



#ifdef USE_MINIBATCH 
                fc_nn_end_block.backpropagtion();
                fc_nn_end_block.update_all_weights(0);
#else
                fc_nn_end_block.clear_batch_accum();
                fc_nn_end_block.backpropagtion();
                conv_L3.clear_kernel_delta();
                conv_L2.clear_kernel_delta();
                conv_L1.clear_kernel_delta();
                conv_L3.clear_i_tens_delta();
                conv_L2.clear_i_tens_delta();
                conv_L1.clear_i_tens_delta();
#endif
                // backprop convolution layers
                // TODO reload f image to convolution input tenor to make a proper backprop of convolution layers
                for (int f = 0; f < nr_frames_strobed; f++)
                {
                    for (int oc = 0; oc < L3_out_ch; oc++)
                    {
                        for (int yi = 0; yi < L3_out_one_side; yi++)
                        {
                            for (int xi = 0; xi < L3_out_one_side; xi++)
                            {
                                conv_L3.o_tensor_delta[oc][yi][xi] = fc_nn_end_block.i_layer_delta[f * oc * L3_out_one_side * L3_out_one_side + oc * L3_out_one_side * L3_out_one_side + yi * L3_out_one_side + xi];
                            }
                        }
                    }
                    conv_L1.input_tensor = conv_L1_stored_input_tensor[f];
                    conv_L2.input_tensor = conv_L2_stored_input_tensor[f];
                    conv_L3.input_tensor = conv_L3_stored_input_tensor[f];

                    conv_L3.conv_backprop();
                    conv_L2.o_tensor_delta = conv_L3.i_tensor_delta;
                    conv_L2.conv_backprop();
                    conv_L1.o_tensor_delta = conv_L2.i_tensor_delta;
                    conv_L1.conv_backprop();
#ifndef USE_MINIBATCH 
                    conv_L3.conv_update_weights();
                    conv_L2.conv_update_weights();
                    conv_L1.conv_update_weights();
#endif
                }

#ifdef USE_MINIBATCH 
                if (mini_batch_cnt < mini_batch_size)
                {
                    mini_batch_cnt++;
                }
                else
                {
                    conv_L3.conv_update_weights();
                    conv_L2.conv_update_weights();
                    conv_L1.conv_update_weights();
                    fc_nn_end_block.update_all_weights(1);
             //       cout << "=========================================" << endl;
             //       cout << "======== Policy network updated =========" << endl;
             //       cout << "=========================================" << endl;
                    fc_nn_end_block.clear_batch_accum();
                    conv_L3.clear_kernel_delta();
                    conv_L2.clear_kernel_delta();
                    conv_L1.clear_kernel_delta();
                    conv_L3.clear_i_tens_delta();
                    conv_L2.clear_i_tens_delta();
                    conv_L1.clear_i_tens_delta();
                    mini_batch_cnt = 0;
                }
#else
                fc_nn_end_block.update_all_weights(1);
#endif
                if (update_frz_cnt < update_frozen_after_samples)
                {
                    update_frz_cnt++;
                }
                else
                {
                    update_frz_cnt = 0;
                    // copy over to frozen fc and conv network
                    conv_frozen_L1_target_net.kernel_weights = conv_L1.kernel_weights;
                    conv_frozen_L2_target_net.kernel_weights = conv_L2.kernel_weights;
                    conv_frozen_L3_target_net.kernel_weights = conv_L3.kernel_weights;
                    fc_nn_frozen_target_net.all_weights = fc_nn_end_block.all_weights;
       //             cout << "=========================================" << endl;
       //             cout << "======== Target network updated =========" << endl;
       //             cout << "=========================================" << endl;
                    // Show upsampling
                    // Put in the output data from the convolution operation into the transpose upsampling operation

                    conv_L3.o_tensor_delta = conv_L3.output_tensor;
                    conv_L3.conv_transpose_fwd();
                    conv_L2.o_tensor_delta = conv_L2.output_tensor;
                    conv_L2.conv_transpose_fwd();
                    conv_L1.o_tensor_delta = conv_L2.i_tensor_delta;
                    conv_L1.conv_transpose_fwd();

                    // Copy data from conv_L1.i_tensor_delta to cv::Mat
                    for (int ic = 0; ic < L1_input_channels; ic++)
                    {
                        for (int yi = 0; yi < pixel_height; yi++)
                        {
                            for (int xi = 0; xi < pixel_width; xi++)
                            {
                                double input_pixel_data = conv_L1.i_tensor_delta[ic][yi][xi];
                                upsampl_conv_view.at<float>(ic * pixel_height + yi, xi) = (float)input_pixel_data;
                                // cout << "input_pixel_data = " << input_pixel_data << endl;
                            }
                        }
                    }
                    cv::imshow("upsampl_conv_view", upsampl_conv_view);
                    waitKey(1);
                    upsampl_conv_view_2 = upsampl_conv_view + 0.5;
                    cv::imshow("upsampl_conv_view_2", upsampl_conv_view_2);
                    waitKey(1);

                }

  //              if (g_replay_nr == 0 && single_game_frame_state == single_game_state_size - 1)
  //              {
  //              }
            }
            cout << "                                                                                                       " << endl;
            std::cout << "\033[F";
            cout <<"g_replay_state_cnt count down = " << (single_game_state_size * g_replay_size) - g_replay_state_cnt << endl;
            // Move the cursor up one line (ANSI escape code)
            std::cout << "\033[F";
        }
     //   imshow("replay_grapics_buffert", replay_grapics_buffert);
     //   waitKey(1);

        // Save all weights
        if (save_cnt < save_after_nr)
        {
            save_cnt++;
        }
        else
        {
            save_cnt = 0;
            fc_nn_end_block.save_weights(weight_filename_end);
            conv_L1.save_weights(L1_kernel_k_weight_filename, L1_kernel_b_weight_filename);
            conv_L2.save_weights(L2_kernel_k_weight_filename, L2_kernel_b_weight_filename);
            conv_L3.save_weights(L3_kernel_k_weight_filename, L3_kernel_b_weight_filename);
        }
        // End
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
