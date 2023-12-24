
#include <iostream>
#include <stdio.h>
#include "fc_m_resnet.hpp"
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

#define USE_MINIBATCH
#define EPISODE_RANDOM_REPLAY
// #define TIMESTEP_RANDOM_REPLAY
#define Q_ALGORITHM_MODE_A

vector<int> fisher_yates_shuffle(vector<int> table);

int main()
{
    srand(static_cast<unsigned>(time(NULL))); // Seed the randomizer
    cout << "Convolution neural network under work..." << endl;
    int total_plays = 0;
    /*
        // ======== create 2 convolution layer objects =========
        convolution conv_L1;
        convolution conv_L2;
        convolution conv_L3;
        convolution conv_frozen_L1_target_net;
        convolution conv_frozen_L2_target_net;
        convolution conv_frozen_L3_target_net;
    */
    //======================================================
    pinball_game gameObj1;      /// Instaniate the pinball game
    gameObj1.init_game();       /// Initialize the pinball game with serten parametrers
    gameObj1.slow_motion = 0;   /// 0=full speed game. 1= slow down
    gameObj1.replay_times = 0;  /// If =0 no replay. >0 this is the nuber of replay with serveral diffrent actions so the ageint take the best rewards before make any weights update
    gameObj1.advanced_game = 0; /// 0= only a ball. 1= ball give awards. square gives punish
    gameObj1.use_image_diff = 0;
    gameObj1.high_precition_mode = 0; /// This will make adjustable rewards highest at center of the pad.
    gameObj1.use_dice_action = 0;
    gameObj1.drop_out_percent = 0;
    gameObj1.Not_dropout = 1;
    gameObj1.flip_reward_sign = 0;
    gameObj1.print_out_nodes = 0;
    gameObj1.enable_ball_swan = 0;
    gameObj1.use_character = 0;
    gameObj1.enabel_3_state = 1; // Input Action from Agent. move_up: 1= Move up pad. 0= Move down pad. 2= STOP used only when enabel_3_state = 1

    // Set up a OpenCV mat
    const int pixel_height = 30; /// The input data pixel height, note game_Width = 220
    const int pixel_width = 30;  /// The input data pixel width, note game_Height = 200
    Mat resized_grapics, replay_grapics_buffert, game_video_full_size, upsampl_conv_view;
    Mat input_frm;

    Size image_size_reduced(pixel_height, pixel_width); // the dst image size,e.g.50x50

    vector<vector<int>> replay_actions_buffert;

    vector<vector<double>> rewards_at_game_replay;

    const int nr_frames_strobed = 4; // 4 Images in serie to make neural network to see movments
    //=========== Neural Network size settings ==============
    fc_m_resnet fc_nn_end_block;
    fc_m_resnet fc_nn_frozen_target_net;
    int save_cnt = 0;

    string weight_filename_end;
    weight_filename_end = "end_block_weights.dat";

    const int all_clip_der = 0;

    fc_nn_end_block.get_version();
    fc_nn_end_block.block_type = 2;
    fc_nn_end_block.use_softmax = 0;                         // 0= Not softmax for DQN reinforcement learning
    fc_nn_end_block.activation_function_mode = 2;            // ReLU for all fully connected activation functions except output last layer
    fc_nn_end_block.force_last_activation_function_mode = 3; // 1 = Last output last layer will have Sigmoid functions regardless mode settings of activation_function_mode
    fc_nn_end_block.use_skip_connect_mode = 0;               // 1 for residual network architetcture
    fc_nn_end_block.use_dropouts = 0;
    fc_nn_end_block.dropout_proportion = 0.0;
    fc_nn_end_block.clip_deriv = all_clip_der;

    fc_nn_frozen_target_net.block_type = fc_nn_end_block.block_type;
    fc_nn_frozen_target_net.use_softmax = fc_nn_end_block.use_softmax;
    fc_nn_frozen_target_net.force_last_activation_function_mode = fc_nn_end_block.force_last_activation_function_mode;
    fc_nn_frozen_target_net.activation_function_mode = fc_nn_end_block.activation_function_mode;
    fc_nn_frozen_target_net.use_skip_connect_mode = fc_nn_end_block.use_skip_connect_mode;
    fc_nn_frozen_target_net.use_dropouts = 0;
    fc_nn_frozen_target_net.clip_deriv = all_clip_der;
    // output channels
    int end_inp_nodes = pixel_height * pixel_width * nr_frames_strobed;
    cout << "end_inp_nodes = " << end_inp_nodes << endl;
    const int end_hid_layers = 3;
    const int end_hid_nodes_L1 = 200;
    const int end_hid_nodes_L2 = 100;
    const int end_hid_nodes_L3 = 30;
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

    double target_off_level = 0.5; // OFF action target. 0.0 you Need to use force_last_activation_function_mode = 3
    double reward_gain = 1.0;
    const double learning_rate_fc = 0.000001;
    double learning_rate_end = learning_rate_fc;
    fc_nn_end_block.learning_rate = learning_rate_end;
#ifdef USE_MINIBATCH
    fc_nn_end_block.momentum = 1.0; // 1.0 for batch fc backpropagation
#else
    fc_nn_end_block.momentum = 0.98; //
#endif
    double init_random_weight_propotion = 0.6;
    const double warm_up_epsilon_start = 0.85;
    double warm_up_epsilon = warm_up_epsilon_start;
    const double warm_up_eps_derating = 0.15;
    const int warm_up_eps_nr = 3;
    int warm_up_eps_cnt = 0;
    const double start_epsilon = 0.50;
    const double stop_min_epsilon = 0.2;
    const double derating_epsilon = 0.001;
    double dqn_epsilon = start_epsilon; // Exploring vs exploiting parameter weight if dice above this threshold chouse random action. If dice below this threshold select strongest outoput action node
    if (warm_up_eps_nr > 0)
    {
        dqn_epsilon = warm_up_epsilon;
    }
    double gamma = 0.8f;
#ifndef Q_ALGORITHM_MODE_A
    double alpha = 0.8;
#endif
    const int g_replay_size = 1000; // Should be 10000 or more
    const int retraing_times = 1;
    const int save_after_nr = 1;
    int update_frz_cnt = 0;
    // statistics report
    const int max_w_p_nr = 1000;
    int win_p_cnt = 0;
    int win_counter = 0;
    double last_win_probability = 0.5;
    double now_win_probability = last_win_probability;
#ifdef USE_MINIBATCH
    const int mini_batch_size = 32;
    int mini_batch_cnt = 0;
#endif
    const int update_frozen_after_samples = 32 * 8;

    //==== Hyper parameter settings End ===========================

    int g_replay_nr = 0;                                                    // Used during play
    int single_game_state_size = gameObj1.nr_of_frames - nr_frames_strobed; // the first for frames will not have any state
    cout << " single_game_state_size = " << single_game_state_size << endl;
#ifdef EPISODE_RANDOM_REPLAY
    cout << "Full random replay mode " << endl;
    vector<int> check_g_replay_list;
    for (int i = 0; i < (single_game_state_size * g_replay_size); i++)
    {
        check_g_replay_list.push_back(0);
    } // Used during replay training
#endif
    vector<int> g_replay_state_rand_list;
    for (int j = 0; j < single_game_state_size; j++)
    {
        g_replay_state_rand_list.push_back(0);
    }

    char answer;
    cout << "Do you want to load kernel weights from saved weight file = Y/N " << endl;
    cin >> answer;
    if (answer == 'Y' || answer == 'y')
    {
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
    }
    fc_nn_frozen_target_net.all_weights = fc_nn_end_block.all_weights;
    cout << "gameObj1.gameObj1.game_Height " << gameObj1.game_Height << endl;
    cout << "gameObj1.gameObj1.game_Width " << gameObj1.game_Width << endl;

    // Start onlu one game now, only for get out size of grapichs to prepare memory
    gameObj1.replay_episode = 0;
    gameObj1.start_episode();

    game_video_full_size = gameObj1.gameGrapics.clone();
    resize(game_video_full_size, resized_grapics, image_size_reduced);
    imshow("resized_grapics", resized_grapics); ///  resize(src, dst, size);
    int replay_row_size = pixel_height * gameObj1.nr_of_frames;
    int replay_col_size = pixel_width * g_replay_size;
    replay_grapics_buffert.create(replay_row_size, replay_col_size, CV_32FC1);
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
        replay_actions_buffert.push_back(dummy_1D_vect_int);    // Create the size of replay_action_buffert memory how store replay action of all states and the whole batch
        rewards_at_game_replay.push_back(dummy_1D_vect_double); //
    }
    const int max_nr_epochs = 1000000;
    for (int epoch = 0; epoch < max_nr_epochs; epoch++)
    {
        cout << "******** Epoch number = " << epoch << " **********" << endl;
        cout << "dqn_epsilon = " << dqn_epsilon << endl;
        for (int g_replay_cnt = 0; g_replay_cnt < g_replay_size; g_replay_cnt++)
        {
            g_replay_nr = g_replay_cnt;
            gameObj1.start_episode();
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
                    for (int i = 0; i < end_inp_nodes; i++)
                    {
                        int row = i / pixel_width;
                        int col = i % pixel_width;
                        int replay_column = col + pixel_width * g_replay_nr;
                        int replay_row = pixel_height * (frame_g - (nr_frames_strobed - 1)) + row;
                        float pixelValue = replay_grapics_buffert.at<float>(replay_row, replay_column);
                        fc_nn_frozen_target_net.input_layer[i] = pixelValue;
                    }

                    //**********************************************************************
                    //****************** Forward Pass training network *********************
                    //**********************************************************************
                    fc_nn_frozen_target_net.forward_pass();                 // Forward pass though fully connected network
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
                            double action_node = fc_nn_frozen_target_net.output_layer[i];
                            if (action_node > max_decision)
                            {
                                max_decision = action_node;
                                decided_action = i;
                            }
                            if (g_replay_cnt < 1)
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
            if (abs_diff < 1.0)
            {
                abs_diff = 1.0;
            }

            if (gameObj1.win_this_game == 1)
            {
                if (gameObj1.square == 1)
                {

                    rewards = reward_gain * 0.5; // Win Rewards avoid square
                                                 //       rewards /= abs_diff;
                }
                else
                {
                    rewards = reward_gain * 1.0; // Win Rewards catch ball
                                                 //       rewards /= abs_diff;
                }
                win_counter++;
            }
            else
            {
                if (gameObj1.square == 1)
                {
                    //  rewards = -2.35; // Lose Penalty
                    rewards = reward_gain * (-1.0);
                    // rewards /= abs_diff;
                }
                else
                {
                    // rewards = -3.95; // Lose Penalty
                    rewards = reward_gain * (-0.5);
                    // rewards *= abs_diff;
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
        // g_replay_nr
        double loss_report = 0.0;
        double min_loss = 9999999999999999999.0;
        double max_loss = 0.0;
        double avg_loss = 0.0;
        int loss_update_cnt = 0;
#ifdef EPISODE_RANDOM_REPLAY
        int g_replay_nr = 0;
        //******************** Go through the batch of replay memory *******************
        cout << "********************************************************************************" << endl;
        cout << "********* Run the whole replay batch memory and training the DQN network *******" << endl;
        cout << "********************************************************************************" << endl;
#ifdef TIMESTEP_RANDOM_REPLAY
        cout << "Training full randomize replay both time step and episode randomize ..." << endl;
#else
        cout << "Training episode randomize but timestep run in series backwards ..." << endl;
#endif
        for (int rt = 0; rt < retraing_times; rt++)
        {
            min_loss = 9999999999999999999.0;
            //   cout << "single_game_state_size = " << single_game_state_size << endl;
            // g_replay_state_rand_list = fisher_yates_shuffle(g_replay_state_rand_list);
            {
                int replay_decided_action = 0;
                fc_nn_end_block.clear_batch_accum();
                check_g_replay_list = fisher_yates_shuffle(check_g_replay_list);

                for (int g_replay_state_cnt = 0; g_replay_state_cnt < (single_game_state_size * g_replay_size); g_replay_state_cnt++)
                {
                    //     cout << "Run one training state sample at replay memory at check_state_nr = " << check_state_nr << endl;
                    g_replay_nr = check_g_replay_list[g_replay_state_cnt] / single_game_state_size;
                    //    cout << "Run one training state sample at g_replay_nr = " << g_replay_nr << endl;
#ifdef TIMESTEP_RANDOM_REPLAY
                    int single_game_frame_state = check_g_replay_list[g_replay_state_cnt] % single_game_state_size;
#else
                    int single_game_frame_state = single_game_state_size - 1 - (g_replay_state_cnt % single_game_state_size);
#endif

#else
        for (int g_replay_nr = 0; g_replay_nr < g_replay_size; g_replay_nr++)
        {
            //******************** Go through the batch of replay memory *******************
            if (g_replay_nr == 0)
            {
                cout << "********************************************************************************" << endl;
                cout << "********* Run the whole replay batch memory and training the DQN network *******" << endl;
                cout << "********************************************************************************" << endl;
#ifdef TIMESTEP_RANDOM_REPLAY
                cout << "Training randomize replay time step but throue full episode in a serie ..." << endl;
#else
                cout << "Training timestep run in series backwards and throue full episode in a serie..." << endl;
#endif
            }

            //   cout << "single_game_state_size = " << single_game_state_size << endl;
            // g_replay_state_rand_list = fisher_yates_shuffle(g_replay_state_rand_list);
            int replay_decided_action = 0;
            fc_nn_end_block.clear_batch_accum();
            for (int rt = 0; rt < retraing_times; rt++)
            {
                g_replay_state_rand_list = fisher_yates_shuffle(g_replay_state_rand_list);
                for (int g_replay_state_cnt = 0; g_replay_state_cnt < single_game_state_size; g_replay_state_cnt++)
                {
#ifdef TIMESTEP_RANDOM_REPLAY
                    int single_game_frame_state = g_replay_state_rand_list[g_replay_state_cnt];
#else
                    int single_game_frame_state = single_game_state_size - 1 - g_replay_state_cnt;
#endif

#endif
                    //    cout << "single_game_frame_state = " << single_game_frame_state << endl;
                    double max_Q_target_value = 0.0;
                    // Calculate the starting column index for the ROI in replay_grapics_buffert
                    for (int i = 0; i < end_inp_nodes; i++)
                    {
                        int row = i / pixel_width;
                        int col = i % pixel_width;
                        float pixelValue = replay_grapics_buffert.at<float>(pixel_height * single_game_frame_state + row, col + pixel_width * g_replay_nr);
                        fc_nn_end_block.input_layer[i] = pixelValue;
                    }
                    //**********************************************************************
                    //****************** Forward Pass training network *********************

                    fc_nn_end_block.forward_pass(); // Forward pass though fully connected network

                    //****************** Forward Pass training network complete ************
                    //**********************************************************************
                    replay_decided_action = replay_actions_buffert[single_game_frame_state + nr_frames_strobed - 1][g_replay_nr];

                    //======================================================================
                    //================== Forward Pass Frozen network NEXT state ============
                    single_game_frame_state++; // Take NEXT state to peak into and get next state Q-value for a target value to train on
                    // Calculate the starting column index for the ROI in replay_grapics_buffert
                    for (int i = 0; i < end_inp_nodes; i++)
                    {
                        int row = i / pixel_width;
                        int col = i % pixel_width;
                        float pixelValue = replay_grapics_buffert.at<float>(pixel_height * single_game_frame_state + row, col + pixel_width * g_replay_nr);
                        fc_nn_frozen_target_net.input_layer[i] = pixelValue;
                    }
                    // Start Forward pass fully connected network
                    fc_nn_frozen_target_net.forward_pass(); // Forward pass though fully connected network
                    //================== Forward Pass Frozen network complete ==============
                    //======================================================================

                    int rewards_idx_state = single_game_frame_state + nr_frames_strobed - 1;
                    double rewards_here = rewards_at_game_replay[rewards_idx_state][g_replay_nr];
                    // cout << "single_game_frame_state = " << single_game_frame_state << endl;
                    // cout << "rewards_idx_state = " << rewards_idx_state << endl;
                    // cout << "rewards_here = " << rewards_here << endl;

                    // Search for max Q-value
                    max_Q_target_value = 0.0;
                    if (single_game_frame_state < single_game_state_size)
                    {
                        for (int i = 0; i < end_out_nodes; i++)
                        {
                            double action_node = fc_nn_frozen_target_net.output_layer[i];
                            if (action_node > max_Q_target_value)
                            {
                                max_Q_target_value = action_node;
                            }
                        }
                    }
                    else
                    {
                    //    cout << "single_game_frame_state = " << single_game_frame_state << endl;
                    //    cout << "rewards_here = " << rewards_here << endl;
                        max_Q_target_value = target_off_level;
                    }

                    if (rewards_here != 0.0)
                    {
                        //           cout << "rewards_here = " << rewards_here << " at rewards_idx_state = " << rewards_idx_state << " at g_replay_nr = " << g_replay_nr << endl;
                    }
                    //     double target_value = rewards_here + gamma * max_Q_target_value;
                    //        #Q table UPDATE
                    //        Q[state,action] = Q[state,action] + ALPHA * (reward + GAMMA * np.max(Q[state_next,:]) - Q[state,action])
                    //   double target_value = rewards_here + gamma * (max_Q_target_value - );
                    // decided_action

                    for (int i = 0; i < end_out_nodes; i++)
                    {
                        if (replay_decided_action == i)
                        {
#ifdef Q_ALGORITHM_MODE_A
                            // target_value = rewards_here + gamma * (max_Q_target_value - );
                            fc_nn_end_block.target_layer[i] = rewards_here + gamma * max_Q_target_value;
#else
                            // Q[state,action] = Q[state,action] + ALPHA * (reward + GAMMA * np.max(Q[state_next,:]) - Q[state,action])
                            fc_nn_end_block.target_layer[i] = fc_nn_end_block.target_layer[i] + alpha * (rewards_here + gamma * max_Q_target_value - fc_nn_end_block.target_layer[i]);
#endif
                        }
                        else
                        {
                            fc_nn_end_block.target_layer[i] = target_off_level;

                            // fc_nn_end_block.target_layer[i] = fc_nn_end_block.target_layer[i]; // No change
                        }
                    }

                    // fc_nn_end_block.backpropagtion_and_update();

#ifdef USE_MINIBATCH
                    fc_nn_end_block.backpropagtion();
                    fc_nn_end_block.update_all_weights(0);
#else
                    fc_nn_end_block.clear_batch_accum();
                    fc_nn_end_block.backpropagtion();
#endif

#ifdef USE_MINIBATCH
                    if (mini_batch_cnt < mini_batch_size)
                    {
                        mini_batch_cnt++;
                    }
                    else
                    {
                        fc_nn_end_block.update_all_weights(1);
                        //       cout << "=========================================" << endl;
                        //       cout << "======== Policy network updated =========" << endl;
                        //       cout << "=========================================" << endl;
                        fc_nn_end_block.clear_batch_accum();
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
                        fc_nn_frozen_target_net.all_weights = fc_nn_end_block.all_weights;
                        //             cout << "=========================================" << endl;
                        //             cout << "======== Target network updated =========" << endl;
                        //             cout << "=========================================" << endl;
                        loss_report = fc_nn_end_block.loss_B;
                        avg_loss += loss_report; // add upp loss
                        loss_update_cnt++;
                        fc_nn_end_block.loss_B = 0.0;
                        if (min_loss > loss_report)
                        {
                            min_loss = loss_report;
                        }
                        if (max_loss < loss_report)
                        {
                            max_loss = loss_report;
                        }
                    }

                    cout << "                                                                                                       " << endl;
                    std::cout << "\033[F";
                    int count_down = 0;
#ifdef EPISODE_RANDOM_REPLAY
                    count_down = (single_game_state_size * g_replay_size) - g_replay_state_cnt;
                    cout << "rt = " << rt << " count down = " << count_down << "  loss = " << loss_report << " min_loss = " << min_loss << endl;
#else
                    count_down = single_game_state_size - g_replay_state_cnt;
                    cout << "g_replay_nr = " << g_replay_nr << " rt = " << rt << "  g_replay_state_cnt count down = " << count_down << "  loss = " << loss_report << " min_loss = " << min_loss << endl;
#endif
                    // Move the cursor up one line (ANSI escape code)
                    std::cout << "\033[F";
                }
            }
#ifdef EPISODE_RANDOM_REPLAY
            avg_loss = avg_loss / loss_update_cnt;
            cout << endl;
            cout << "***********************" << endl;
            cout << "Avarage loss = " << avg_loss << endl;
            cout << "***********************" << endl;
        }
#else
        }
        avg_loss = avg_loss / loss_update_cnt;
        cout << endl;
        cout << "***********************" << endl;
        cout << "Avarage loss = " << avg_loss << endl;
        cout << "***********************" << endl;
        avg_loss = 0.0;

#endif
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
