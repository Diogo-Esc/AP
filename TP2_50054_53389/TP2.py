#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aprendizagem Profunda, TP1
Diogo Escaleira 50054
João Azevedo 53389
"""

from snake_game import SnakeGame
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

ACTIONS = [-1,0,1]

WIDTH_HEIGHT = 14

MIN_REPLAY_SIZE = 1000
MEMORY_MAX_LEN = 100000
EPSILON = 0.05
MAX_EPSILON = 1
MIN_EPSILON = 0.01
DECAY = 0.01
N_EPISODES = 1000

DISCOUNT_FACTOR = 0.9
BATCH_SIZE = 1000

def plot_board(file_name, board, text=None):
    plt.figure(figsize=(10,10))
    plt.imshow(board)
    plt.axis('off')
    if text is not None:
        plt.gca().text(3, 3, text, fontsize=45,color = 'yellow')
    plt.savefig(file_name,bbox_inches='tight')
    plt.close()

def plot_steps(file_name, steps):
    plt.plot(range(len(steps)), steps)
    plt.grid(linestyle = '--')
    plt.suptitle(file_name)
    plt.ylabel('N Steps')
    plt.xlabel('Episode')
    plt.savefig('snake_steps.png')
    plt.close()

def plot_scores(file_name, scores):
    plt.plot(range(len(scores)), scores)
    plt.grid(linestyle = '--')
    plt.suptitle(file_name)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.savefig('snake_scores.png')
    plt.close()


def get_games(memory, snake_game):
    #A0 - turn left (-1)
    #A1 - keep straight (0)
    #A2 - turn right (1)
    A0 = 0
    A1 = 0
    A2 = 0
    score = 0
    while(len(memory) < MEMORY_MAX_LEN):
        bol = False
        board_state, reward, done, score = snake_game.reset()

        n_steps = 0
        r_nula, r_positiva, r_negativa = 0,0,0
        
        while(not done):

            score, apples, head, tail, direction = snake_game.get_state()
        
            #(row, col)
            target_loc = (apples[0][0], apples[0][1])
            a_h_col_dif = target_loc[1] - head[1]
            a_h_row_dif = target_loc[0] - head[0]

            action = 0

            if(direction == 2):
                if(a_h_row_dif > 0):
                    action = 0
                    A1 += 1
                elif(a_h_row_dif == 0):
                    if(a_h_col_dif < 0 and not (tail[1] == head[1]+1 and tail[0] == head[0])):
                        action = 1
                        A2 += 1
                    elif(a_h_col_dif > 0 and not (tail[1] == head[1]-1 and tail[0] == head[0])):
                        action = -1
                        A0 += 1
                elif(a_h_row_dif < 0):
                    if (head[1] < WIDTH_HEIGHT/2 and not (tail[1] == head[1]-1 and tail[0] == head[0])):
                        action = -1
                        A0 += 1
                    elif (head[1] >= WIDTH_HEIGHT/2 and not (tail[1] == head[1]+1 and tail[0] == head[0])):
                        action = 1
                        A2 += 1

            elif(direction == 0):
                if(a_h_row_dif < 0):
                    action = 0
                    A1 += 1
                elif(a_h_row_dif == 0):
                    if(a_h_col_dif < 0 and not (tail[1] == head[1]+1 and tail[0] == head[0])):
                        action = -1
                        A0 += 1
                    elif(a_h_col_dif > 0 and not (tail[1] == head[1]-1 and tail[0] == head[0])):
                        action = 1
                        A2 += 1
                elif(a_h_row_dif > 0):
                    if (head[1] < WIDTH_HEIGHT/2 and not (tail[1] == head[1]-1 and tail[0] == head[0])):
                        action = 1
                        A2 += 1
                    elif (head[1] >= WIDTH_HEIGHT/2 and not (tail[1] == head[1]+1 and tail[0] == head[0])):
                        action = -1
                        A0 += 1

            elif(direction == 3):
                if(a_h_col_dif < 0):
                    action = 0
                    A1 += 0
                elif(a_h_col_dif == 0):
                    if(a_h_row_dif < 0 and not (tail[0] == head[0]-1 and tail[1] == head[1])):
                        action = 1
                        A2 += 1
                    elif(a_h_row_dif > 0 and not (tail[0] == head[0]+1 and tail[1] == head[1])):
                        action = -1
                        A0 += 1
                elif(a_h_col_dif > 0):
                    if(head[0] > WIDTH_HEIGHT/2 and not (tail[0] == head[0]-1 and tail[1] == head[1])):
                        action = 1
                        A2 += 1
                    elif(head[0] <= WIDTH_HEIGHT/2 and not (tail[0] == head[0]+1 and tail[1] == head[1])):
                        action = -1
                        A0 += 1

            else:
                if(a_h_col_dif > 0):
                    action = 0
                    A1 += 1
                elif(a_h_col_dif == 0):
                    if(a_h_row_dif < 0 and not (tail[0] == head[0]-1 and tail[1] == head[1])):
                        action = -1
                        A0 += 1
                    elif(a_h_row_dif > 0 and not (tail[0] == head[0]+1 and tail[1] == head[1])):
                        action = 1
                        A2 += 1
                elif(a_h_col_dif < 0):
                    if (head[0] < WIDTH_HEIGHT/2 and not (tail[0] == head[0]+1 and tail[1] == head[1])):
                        action = 1
                        A2 += 1
                    elif (head[0] >= WIDTH_HEIGHT/2 and not (tail[0] == head[0]-1 and tail[1] == head[1])):
                        action = -1
                        A0 += 1

            new_board_state, reward, done, score = snake_game.step(action)

            if(reward < 0):
                r_negativa += 1
            elif(reward == 0):
                r_nula += 1
            else:
                r_positiva += 1
            
            score = r_positiva - r_negativa
            n_steps += 1

            #Estado do tabuleiro antes da acao, acao, recompensa, estado do novo tabuleiro apos acao, se chegou a um estado final
            memory.append([board_state, action, reward, new_board_state, done])
            board_state = new_board_state
        print("--------------------------#---------------------------#--------------------------")
        print("Rewards Positivas ", r_positiva, "; Rewards Negativas ", r_negativa, "; Rewards Nulas ", r_nula, "; Balanco ", score)
        print("Acoes Esquerda ", A0, "; Acoes Frente ", A1, "; Acoes Direita ", A2)
        print("Numero de Steps ", n_steps)
        print("Memoria Atingida: ", len(memory),"de", memory.maxlen)


def agent(state_shape, action_shape):

    learning_rate = 0.001

    init = tf.keras.initializers.HeUniform()

    inputs = keras.layers.Input(shape=state_shape, name='inputs')

    layer = keras.layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer = init)(inputs)
    layer = keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)
    layer = keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer = init)(layer)
    layer = keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer = init)(layer)
    layer = keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)

    layer = keras.layers.Flatten()(layer)

    layer = keras.layers.Dense(64, activation='relu', kernel_initializer = init)(layer)
    outputs = keras.layers.Dense(action_shape,
                                 activation='linear', kernel_initializer = init)(layer)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
    return model

def train_agent(environment, replay_memory, model, target_model, done):
    batch_size = BATCH_SIZE
    mini_batch = random.sample(replay_memory, batch_size)

    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)

    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)
    
    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + DISCOUNT_FACTOR * np.max(future_qs_list[index])
        else:
            max_future_q = reward
        current_qs = current_qs_list[index]
        current_qs[action] = max_future_q
        X.append(observation)
        Y.append(current_qs)
    
    return model.fit(np.array(X), np.array(Y), batch_size = batch_size, verbose = 0, shuffle = True)


def run():
    snake_game = SnakeGame(WIDTH_HEIGHT, WIDTH_HEIGHT, grass_growth=0.001, max_grass=0.05, border=1)

    replay_memory = deque(maxlen=MEMORY_MAX_LEN)
    get_games(replay_memory, snake_game)

    model = agent(snake_game.board_state().shape, len(ACTIONS))
    target_model = agent(snake_game.board_state().shape, len(ACTIONS))
    target_model.set_weights(model.get_weights())

    epsilon=1
    decay=0.004
    steps_to_update = 0
    scores = []
    steps = []
    for i in range(N_EPISODES):
        n_steps = 0
        rewards = 0
        done = False
        is_random = 0

        board_state, reward, done, score = snake_game.reset()

        while(not done):
            
            rand = np.random.rand()
            steps_to_update += 1
            n_steps+=1

            if(is_random == 1):
                plot_board('snake_game_episode{}_step{}'.format(i, n_steps), board_state, "Random")
            else: 
                plot_board('snake_game_episode{}_step{}'.format(i, n_steps), board_state, "Model")

            if (rand <= epsilon):
                action = np.random.choice(ACTIONS) 
                is_random = 1
            
            else:
                reshaped_board = board_state.reshape([1, board_state.shape[0], board_state.shape[1], board_state.shape[2]])
                pred = model.predict(reshaped_board).flatten()
                action = np.argmax(pred) -1
                print("Action Escolhida ", action)
                is_random = 0

            new_board_state, reward, done, score = snake_game.step(action)  

            rewards += reward

            replay_memory.append([board_state, action, reward, new_board_state, done]) 

            board_state = new_board_state 

            if  (len(replay_memory) >= MIN_REPLAY_SIZE and (steps_to_update % 20 == 0 or done)):
                print("training...")
                trainRes = train_agent(snake_game, replay_memory, model, target_model, done)

            if(done):
                score, apples, head, tail, direction = snake_game.get_state()

                scores.append(score)
                steps.append(n_steps)


                print("Episodio ", i, "; Reward Total", rewards, "; Score total", score, "; Steps", n_steps)
                print("--------------------------#---------------------------#--------------------------")
                if (steps_to_update >= 200):
                    print("updating weights...")
                    target_model.set_weights(model.get_weights())
                    steps_to_update = 0
                
                break
            
            epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-decay * i)

        with writer.as_default():
            tf.summary.scalar("loss", trainRes.history['loss'][0], step=i)

    plot_scores('snake_scores', scores)
    plot_steps('snake_steps', steps)

run()