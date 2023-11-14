
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


vector<int> fisher_yates_shuffle(vector<int> table);

int main()
{
    cout << "Convolution neural network under work..." << endl;
    // ======== create 2 convolution layer objects =========
    convolution conv_L1;
    convolution conv_L2;
    convolution conv_frozen_L1_target_net;
    convolution conv_frozen_L2_target_net;

    //======================================================
    pinball_game gameObj1;///Instaniate the pinball game
    gameObj1.init_game();///Initialize the pinball game with serten parametrers
    gameObj1.slow_motion=0;///0=full speed game. 1= slow down
    gameObj1.replay_times = 0;///If =0 no replay. >0 this is the nuber of replay with serveral diffrent actions so the ageint take the best rewards before make any weights update
    gameObj1.advanced_game = 1;///0= only a ball. 1= ball give awards. square gives punish
    gameObj1.use_image_diff=0;
    gameObj1.high_precition_mode = 1; ///This will make adjustable rewards highest at center of the pad.
    gameObj1.use_dice_action=0;
    gameObj1.drop_out_percent=0;
    gameObj1.Not_dropout=1;
    gameObj1.flip_reward_sign =0;
    gameObj1.print_out_nodes = 0;
    gameObj1.enable_ball_swan =0;
    gameObj1.use_character =0;
    gameObj1.enabel_3_state = 1;//Input Action from Agent. move_up: 1= Move up pad. 0= Move down pad. 2= STOP used only when enabel_3_state = 1

    



    // Set up a OpenCV mat

    const int pixel_height = 50;///The input data pixel height, note game_Width = 220
    const int pixel_width = 50;///The input data pixel width, note game_Height = 200
    Mat resized_grapics, replay_grapics_buffert, game_video_full_size;
    Size image_size_reduced(pixel_height,pixel_width);//the dst image size,e.g.50x50

    vector<vector<int>> replay_actions_buffert;
    
    vector<vector<double>> rewards_at_batch;

    
    //=========== Neural Network size settings ==============
    fc_m_resnet fc_nn_end_block;
    fc_m_resnet fc_nn_frozen_target_net;

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
    fc_nn_end_block.use_softmax = 0;//0= Not softmax for DQN reinforcement learning
    fc_nn_end_block.activation_function_mode = 0;// ReLU for all fully connected activation functions except output last layer
    fc_nn_end_block.force_last_activation_function_to_sigmoid = 0;//1 = Last output last layer will have Sigmoid functions regardless mode settings of activation_function_mode
    fc_nn_end_block.use_skip_connect_mode = 0; // 1 for residual network architetcture
    fc_nn_end_block.use_dropouts = 0;
    fc_nn_end_block.dropout_proportion = 0.4;

    fc_nn_frozen_target_net.block_type = fc_nn_end_block.block_type;
    fc_nn_frozen_target_net.use_softmax = fc_nn_end_block.use_softmax;
    fc_nn_frozen_target_net.force_last_activation_function_to_sigmoid = fc_nn_end_block.force_last_activation_function_to_sigmoid;
    fc_nn_frozen_target_net.use_skip_connect_mode = fc_nn_end_block.use_skip_connect_mode;
    fc_nn_frozen_target_net.use_dropouts = 0;
    

    conv_L1.get_version();

    //==== Set up convolution layers ===========
    cout << "conv_L1 setup:" << endl;
    const int nr_color_channels = 1;//=== 1 channel gray scale ====
    const int nr_frames_strobed = 4;// 4 Images in serie to make neural network to see movments
    const int L1_input_channels = nr_color_channels * nr_frames_strobed;//color channels * Images in serie
    const int L1_tensor_in_size = pixel_width*pixel_height;
    const int L1_tensor_out_channels = 50;
    const int L1_kernel_size = 5;
    const int L1_stride = 3;
    conv_L1.set_kernel_size(L1_kernel_size); // Odd number
    conv_L1.set_stride(L1_stride);
    conv_L1.set_in_tensor(L1_tensor_in_size, L1_input_channels); // data_size_one_sample_one_channel, input channels
    conv_L1.set_out_tensor(L1_tensor_out_channels);                                              // output channels
    conv_L1.top_conv = 1;
    //copy to a frozen copy network for target network
    conv_frozen_L1_target_net.set_kernel_size(L1_kernel_size);
    conv_frozen_L1_target_net.set_stride(L1_stride);
    conv_frozen_L1_target_net.set_in_tensor(L1_tensor_in_size, L1_input_channels);
    conv_frozen_L1_target_net.set_out_tensor(L1_tensor_out_channels);
    conv_frozen_L1_target_net.top_conv = conv_L1.top_conv;
    //========= L1 convolution (vectors) all tensor size for convolution object is finnish =============

    //==== Set up convolution layers ===========
    const int L2_input_channels = conv_L1.output_tensor.size();    
    const int L2_tensor_in_size = (conv_L1.output_tensor[0].size() * conv_L1.output_tensor[0].size());
    const int L2_tensor_out_channels = 50;
    const int L2_kernel_size = 5;
    const int L2_stride = 3;

    cout << "conv_L2 setup:" << endl;
    conv_L2.set_kernel_size(L2_kernel_size); // Odd number
    conv_L2.set_stride(L2_stride);
    conv_L2.set_in_tensor(L2_tensor_in_size, L2_input_channels); // data_size_one_sample_one_channel, input channels
    conv_L2.set_out_tensor(L2_tensor_out_channels);     
    conv_L2.top_conv = 0;
    //copy to a frozen copy network for target network
    conv_frozen_L2_target_net.set_kernel_size(L2_kernel_size);
    conv_frozen_L2_target_net.set_stride(L2_stride);
    conv_frozen_L2_target_net.set_in_tensor(L2_tensor_in_size, L2_input_channels);
    conv_frozen_L2_target_net.set_out_tensor(L2_tensor_out_channels);
    conv_frozen_L2_target_net.top_conv = conv_L2.top_conv;
    
                                                                                              // output channels
    const int end_inp_nodes = (conv_L2.output_tensor[0].size() * conv_L2.output_tensor[0].size()) * conv_L2.output_tensor.size();
    cout << "end_inp_nodes = " << end_inp_nodes << endl;
    const int end_hid_layers = 2;
    const int end_hid_nodes_L1 = 200;
    const int end_hid_nodes_L2 = 100;
    const int end_out_nodes = 3;//Up, Down and Stop action
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
    fc_nn_frozen_target_net.set_nr_of_hidden_layers(end_hid_layers);
    fc_nn_frozen_target_net.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L1);
    fc_nn_frozen_target_net.set_nr_of_hidden_nodes_on_layer_nr(end_hid_nodes_L2);

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
    double init_random_weight_propotion = 0.02;
    double init_random_weight_propotion_conv = 0.01;
    const double start_epsilon = 0.5;
    const double stop_min_epsilon = 0.25;
    const double derating_epsilon = 0.01;//Derating speed per batch game
    double dqn_epsilon = start_epsilon;//Exploring vs exploiting parameter weight if dice above this threshold chouse random action. If dice below this threshold select strongest outoput action node
    double gamma = 0.91f;
    //==== Hyper parameter settings End ===========================

    //==== Set modes ===============
    conv_L1.activation_function_mode = 2;
    conv_L2.activation_function_mode = 2;
    conv_frozen_L1_target_net.activation_function_mode = conv_L1.activation_function_mode;
    conv_frozen_L2_target_net.activation_function_mode = conv_L2.activation_function_mode;
    //==============================

    
    const int batch_size = 10;
    int batch_nr = 0;//Used during play
    vector<int> batch_state_rand_list;
    int single_game_state_size = gameObj1.nr_of_frames - nr_frames_strobed + 1;// the first for frames will not have any state
    int check_state_nr = 0;//Used during replay training 
    for(int i=0;i<batch_size;i++)
    {
        for(int j=0;j<single_game_state_size;j++)
        {
            batch_state_rand_list.push_back(0);
        }
    }


    char answer;
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

    srand(static_cast<unsigned>(time(NULL))); // Seed the randomizer
    cout << "gameObj1.gameObj1.game_Height " << gameObj1.game_Height << endl;
    cout << "gameObj1.gameObj1.game_Width " << gameObj1.game_Width << endl;

    //Start onlu one game now, only for get out size of grapichs to prepare memory
    gameObj1.replay_episode = 0;
    gameObj1.start_episode();

    game_video_full_size = gameObj1.gameGrapics.clone();
    resize(game_video_full_size, resized_grapics, image_size_reduced);
    imshow("resized_grapics", resized_grapics); ///  resize(src, dst, size);
    replay_grapics_buffert.create(pixel_height * gameObj1.nr_of_frames , pixel_width * batch_size, CV_32FC1);
    cout << "replay_grapics_buffert rows = " << replay_grapics_buffert.rows << endl;
    cout << "replay_grapics_buffert cols = " << replay_grapics_buffert.cols << endl;
    cout << "resized_grapics rows = " << resized_grapics.rows << endl;
    cout << "resized_grapics cols = " << resized_grapics.cols << endl;

    vector<int> dummy_1D_vect_int;
    vector<double> dummy_1D_vect_double;
    for(int i=0;i<batch_size;i++)
    {
        dummy_1D_vect_int.push_back(0);//Prepare an inner 1D vector to put to next replay_action_buffert 2D vector with size of batch_size
        dummy_1D_vect_double.push_back(0.0);
    }
    for(int i=0;i<gameObj1.nr_of_frames;i++)
    {
        replay_actions_buffert.push_back(dummy_1D_vect_int);//Create the size of replay_action_buffert memory how store replay action of all states and the whole batch
        rewards_at_batch.push_back(dummy_1D_vect_double);//
    }

    const int max_nr_epochs = 1000000;
    for (int epoch = 0; epoch < max_nr_epochs; epoch++)
    {
        for (int batch_cnt=0; batch_cnt < batch_size; batch_cnt++)
        {
            batch_nr = batch_cnt;
            gameObj1.start_episode();
            cout << "Run one game and store it in replay memory index at batch_cnt = " << batch_cnt << endl;
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
                int startCol = pixel_width * batch_nr;
                // Create a Rect to define the ROI in replay_grapics_buffert
                cv::Rect roi(startCol, pixel_height * frame_g, pixel_width, pixel_height);
                // Copy the relevant region from resized_grapics to replay_grapics_buffert
                resized_grapics(cv::Rect(0, 0, pixel_width, pixel_height)).copyTo(replay_grapics_buffert(roi));
                if (frame_g >= nr_frames_strobed - 1) // Wait until all 4 images is up there in the game after start
                {
                    // Put in data from replay_grapics_buffert to L1_tensor_in_size
                    for (int i = 0; i < L1_input_channels; i++)
                    {
                        for (int j = 0; j < L1_tensor_in_size; j++)
                        {
                            int row = j / pixel_width;
                            int col = j % pixel_width;
                            float pixelValue = replay_grapics_buffert.at<float>(pixel_height * (frame_g - (nr_frames_strobed - 1) + i) + row, col + pixel_width * batch_nr);
                            conv_L1.input_tensor[i][row][col] = pixelValue;
                        }
                    }
                    //**********************************************************************
                    //****************** Forward Pass training network *********************
                    conv_L1.conv_forward1();
                    conv_L2.input_tensor = conv_L1.output_tensor;
                    conv_L2.conv_forward1();
                    int L2_out_one_side = conv_L2.output_tensor[0].size();
                    int L2_out_ch = conv_L2.output_tensor.size();
                    for (int oc = 0; oc < L2_out_ch; oc++)
                    {
                        for (int yi = 0; yi < L2_out_one_side; yi++)
                        {
                            for (int xi = 0; xi < L2_out_one_side; xi++)
                            {
                                fc_nn_end_block.input_layer[oc * L2_out_one_side * L2_out_one_side + yi * L2_out_one_side + xi] = conv_L2.output_tensor[oc][yi][xi];
                               // cout << "conv_L2.output_tensor[oc][yi][xi] = " << conv_L2.output_tensor[oc][yi][xi] << endl;
                            }
                        }
                    }
                    // Start Forward pass fully connected network
                    fc_nn_end_block.forward_pass();// Forward pass though fully connected network
                   
                    float exploring_dice = (float) (rand() % 65535) / 65536;//Through a fair dice. Random value 0..1.0 range
                    //dqn_epsilon = 0.5;//Exploring vs exploiting parameter weight if dice above this threshold chouse random action. If dice below this threshold select strongest outoput action node
                    int decided_action = 0;
                    double max_decision = 0.0f;
                    if(exploring_dice > dqn_epsilon)
                    {
                        //Choose dice action (Exploration mode)
                        for(int i=0;i<end_out_nodes;i++)//end_out_nodes = numbere of actions
                        {
                            float action_dice = (float)(rand() % 65535) / 65536; // Through a fair dice. Random value 0..1.0 range
                            //cout << "action_dice = " << action_dice << endl;
                            if(action_dice > (float)max_decision)
                            {
                                max_decision = (float)action_dice;
                                decided_action = i;
                            }
                        }
                    }
                    else
                    {
                        //Choose predicted action (Exploit mode)
                        for(int i=0;i<end_out_nodes;i++)
                        {
                            //end_out_nodes = numbere of actions
                            double action_node = fc_nn_end_block.output_layer[i];
                            if(action_node > max_decision)
                            {
                                max_decision = action_node;
                                decided_action = i;
                            }
                          //  cout << "action_node = " << action_node << " i = " << i << endl;
                        }

                    }
                  //  std::cout << std::fixed << std::setprecision(20);
                  //  cout << "decided_action = " << decided_action << endl;
                  //  cout << "exploring_dice = " << exploring_dice << endl;

                    gameObj1.move_up = decided_action;//Input Action from Agent. 1= Move up pad. 0= Move down pad. 2= STOP used only when enabel_3_state = 1
                    replay_actions_buffert[frame_g][batch_nr] = decided_action;
                    
                    //****************** Forward Pass training network complete ************
                    //**********************************************************************
                }
            }
            double rewards = 0.0;
            if(gameObj1.win_this_game == 1)
            {
                rewards = 0.5;//Win Rewards
            }
            else
            {
                rewards = -0.5;//Lose Penalty
            }
            rewards_at_batch[gameObj1.nr_of_frames-1][batch_nr] = rewards;

        }
        imshow("replay_grapics_buffert", replay_grapics_buffert);
        waitKey(1000);

        //******************** Go through the batch of replay memory *******************
        cout << "********************************************************************************" << endl;
        cout << "********* Run the whole replay batch memeory and traing the DQN network ********" << endl;
        cout << "********************************************************************************" << endl;
        cout << "single_game_state_size = " << single_game_state_size << endl;
        batch_state_rand_list = fisher_yates_shuffle(batch_state_rand_list);
        int replay_decided_action = 0;
        for (int batch_state_cnt=0; batch_state_cnt < (single_game_state_size * batch_size); batch_state_cnt++)
        {
            check_state_nr = batch_state_rand_list[batch_state_cnt];
            cout << "Run one training state sample at replay memory at check_state_nr = " << check_state_nr << endl;
            batch_nr =  check_state_nr / single_game_state_size;
            cout << "Run one training state sample at batch_nr = " << batch_nr << endl;
            int single_game_frame_state = check_state_nr % single_game_state_size;
            cout << "single_game_frame_state = " << single_game_frame_state << endl;
            double  max_Q_target_value = 0.0; 
            if (single_game_frame_state < single_game_state_size - 1)
            {
                // Calculate the starting column index for the ROI in replay_grapics_buffert
                int startCol = pixel_width * batch_nr;
                int startRow = pixel_height * single_game_frame_state;

                for (int frame_t = 0; frame_t < nr_frames_strobed; frame_t++) // Loop throue 4 frames
                {
                    cv::Rect replay_roi(startCol, (startRow + pixel_height * frame_t), pixel_width, pixel_height);
                    for (int i = 0; i < L1_input_channels; i++)
                    {
                        for (int j = 0; j < L1_tensor_in_size; j++)
                        {
                            int row = j / pixel_width;
                            int col = j % pixel_width;

                            // Calculate the actual row and column indices in replay_roi
                            int roi_row = row + replay_roi.y;
                            int roi_col = col + replay_roi.x;

                            // Ensure the indices are within bounds
                            if (roi_row < replay_roi.y + replay_roi.height && roi_col < replay_roi.x + replay_roi.width)
                            {
                                // Extract replay_roi data here to pixelValue
                                float pixelValue = replay_grapics_buffert.at<float>(roi_row, roi_col);
                                conv_L1.input_tensor[i][row][col] = pixelValue;
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
                }

                //**********************************************************************
                //****************** Forward Pass training network *********************
                conv_L1.conv_forward1();
                conv_L2.input_tensor = conv_L1.output_tensor;
                conv_L2.conv_forward1();
                int L2_out_one_side = conv_L2.output_tensor[0].size();
                int L2_out_ch = conv_L2.output_tensor.size();
                for (int oc = 0; oc < L2_out_ch; oc++)
                {
                    for (int yi = 0; yi < L2_out_one_side; yi++)
                    {
                        for (int xi = 0; xi < L2_out_one_side; xi++)
                        {
                            fc_nn_end_block.input_layer[oc * L2_out_one_side * L2_out_one_side + yi * L2_out_one_side + xi] = conv_L2.output_tensor[oc][yi][xi];
                        }
                    }
                }
                // Start Forward pass fully connected network
                fc_nn_end_block.forward_pass(); // Forward pass though fully connected network


                //****************** Forward Pass training network complete ************
                //**********************************************************************
                replay_decided_action = replay_actions_buffert[single_game_frame_state + nr_frames_strobed-1][batch_nr]; 

                //===================================

                single_game_frame_state++;//Take NEXT state to peak into and get next state Q-value for a target value to train on 
                // Calculate the starting column index for the ROI in replay_grapics_buffert
                startCol = pixel_width * batch_nr;
                startRow = pixel_height * single_game_frame_state;
                for (int frame_t = 0; frame_t < nr_frames_strobed; frame_t++) // Loop throue 4 frames
                {
                    cv::Rect replay_roi(startCol, (startRow + pixel_height * frame_t), pixel_width, pixel_height);
                    for (int i = 0; i < L1_input_channels; i++)
                    {
                        for (int j = 0; j < L1_tensor_in_size; j++)
                        {
                            int row = j / pixel_width;
                            int col = j % pixel_width;

                            // Calculate the actual row and column indices in replay_roi
                            int roi_row = row + replay_roi.y;
                            int roi_col = col + replay_roi.x;

                            // Ensure the indices are within bounds
                            if (roi_row < replay_roi.y + replay_roi.height && roi_col < replay_roi.x + replay_roi.width)
                            {
                                // Extract replay_roi data here to pixelValue
                                float pixelValue = replay_grapics_buffert.at<float>(roi_row, roi_col);
                                conv_frozen_L1_target_net.input_tensor[i][row][col] = pixelValue;
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
                }

                //======================================================================
                //================== Forward Pass Frozen network NEXT state ============
                conv_frozen_L1_target_net.conv_forward1();
                conv_frozen_L2_target_net.input_tensor = conv_frozen_L1_target_net.output_tensor;
                conv_frozen_L2_target_net.conv_forward1();

                for (int oc = 0; oc < L2_out_ch; oc++)
                {
                    for (int yi = 0; yi < L2_out_one_side; yi++)
                    {
                        for (int xi = 0; xi < L2_out_one_side; xi++)
                        {
                            fc_nn_frozen_target_net.input_layer[oc * L2_out_one_side * L2_out_one_side + yi * L2_out_one_side + xi] = conv_frozen_L2_target_net.output_tensor[oc][yi][xi];
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
                        max_Q_target_value  = action_node;
                    }
                }
            }
            else
            {
                // End game state
                max_Q_target_value = 0.5;//0.5 = Zero Q value at end state Only rewards will be used 
                cout << "Replay END State at batch_nr = " << batch_nr << endl;
            }
            int rewards_idx_state = single_game_frame_state + nr_frames_strobed - 1;
            cout << "rewards_idx_state = " << rewards_idx_state << endl;
            double rewards_here = rewards_at_batch[rewards_idx_state][batch_nr];
            double target_value = rewards_here + gamma * max_Q_target_value;
            
            // decided_action
            for(int i=0;i<end_out_nodes;i++)
            {
                if(i == replay_decided_action)
                {
                    fc_nn_end_block.target_layer[i] = target_value;
                }
                else
                {
                    fc_nn_end_block.target_layer[i] = 0.5;
                }
            }
            
            fc_nn_end_block.backpropagtion_and_update();
            //TODO: backprop convolution layers

            //TODO copy over to frozen fc and conv network 
            
        }
        imshow("replay_grapics_buffert", replay_grapics_buffert);
        waitKey(1);

        //******************** End of replay batch *************************************



    }

    //Save all weights
    fc_nn_end_block.save_weights(weight_filename_end);
    conv_L1.save_weights(L1_kernel_k_weight_filename, L1_kernel_b_weight_filename);
    conv_L2.save_weights(L2_kernel_k_weight_filename, L2_kernel_b_weight_filename);
    //End
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
