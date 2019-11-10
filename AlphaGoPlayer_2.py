from gym import error
try:
    import pachi_py
except ImportError as e:
    # The dependency group [pachi] should match the name is setup.py.
    raise error.DependencyNotInstalled('{}. (HINT: you may need to install the Go dependencies via "pip install gym[pachi]".)'.format(e))

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from six import StringIO
import sys
import six
import goSimi as goSim


def _pass_action(board_size):
    return board_size**2


def _resign_action(board_size):
    return board_size**2 + 1


def _coord_to_action(board, c):
    '''Converts Pachi coordinates to actions'''
    if c == pachi_py.PASS_COORD: return _pass_action(board.size)
    if c == pachi_py.RESIGN_COORD: return _resign_action(board.size)
    i, j = board.coord_to_ij(c)
    return i*board.size + j


def _action_to_coord(board, a):
    '''Converts actions to Pachi coordinates'''
    if a == _pass_action(board.size): return pachi_py.PASS_COORD
    if a == _resign_action(board.size): return pachi_py.RESIGN_COORD
    return board.ij_to_coord(a // board.size, a % board.size)


def str_to_action(board, s):
    return _coord_to_action(board, board.str_to_coord(s.encode()))



class AlphaGoPlayer():
    def __init__(self, init_state, seed, player_color):
        self.init_state=init_state
        self.seed = seed
        self.player_color = player_color
        if(player_color==2):
            para='white'
        else:
            para='black'
        self.board_size = len(init_state[2])
        self.env = goSim.GoEnv(player_color=para, observation_type='image3c', illegal_move_mode="raise", board_size=self.board_size, komi=7.5)
        self.obs_t = self.env.reset()

    def get_action(self, cur_state,opponent_action):
        # Do Coolstuff using cur_state
        # Check illegal Move
        b = self.env.state.board
        obs=b.encode()
        #print("official score",self.env.state.board.official_score)
        #print(self.player_color)
        #print(obs)
        if(opponent_action!=-1):
            self.env.state=self.env.state.act(opponent_action)
        b=self.env.state.board
        isPass = (opponent_action==self.board_size*self.board_size)
        legal_coords = b.get_legal_coords(self.player_color)
        if(len(legal_coords)<20):
            depth=4
        elif(len(legal_coords)<75):
            depth=3
        else:
            depth=2
        print(depth,len(legal_coords))
        val,action = self.minimax(self.env.state,self.player_color,depth,isPass)

        #legal_coords = b.get_legal_coords(self.player_color)
        #for l in legal_coords:
            #print(_coord_to_action(b, l))
        #if(len(legal_coords)<5):
        #    coord=legal_coords[0]
        #else:
        #    coord = legal_coords[4]
        #print(len(legal_coords))
        #action = _coord_to_action(b, coord)
        print('action ',action,self.player_color)
        print("official score",self.env.state.board.official_score)
        self.env.state=self.env.state.act(action)
        return action
    
    
    def getVal(self,env,done):
        if(done):
            if(env.board.official_score + 7.5 < 0):
                return 100000
            elif(env.board.official_score + 7.5 > 0):
                return -100000
            else:
                return 0
        else:
            #s=0
            #obs=env.board.encode()
            #for a_x in range(self.board_size):
            #    for a_y in range(self.board_size):
            #        if(np.all(obs[:, a_x, a_y] == np.array([0,0,1]))):
            #            if(a_x==0 or np.all(obs[:, a_x-1, a_y] == np.array([1,0,0]))):
            #                if(a_x==self.board_size-1 or np.all(obs[:, a_x+1, a_y] == np.array([1,0,0]))):
            #                    if(a_y==0 or np.all(obs[:, a_x, a_y-1] == np.array([1,0,0]))):
            #                        if(a_y==self.board_size-1 or np.all(obs[:, a_x, a_y+1] == np.array([1,0,0]))):
            #                            s+=5
            #            if(a_x==0 or np.all(obs[:, a_x-1, a_y] == np.array([0,1,0]))):
            #                if(a_x==self.board_size-1 or np.all(obs[:, a_x+1, a_y] == np.array([0,1,0]))):
            #                    if(a_y==0 or np.all(obs[:, a_x, a_y-1] == np.array([0,1,0]))):
            #                        if(a_y==self.board_size-1 or np.all(obs[:, a_x, a_y+1] == np.array([0,1,0]))):
            #                            s-=5
            #print(s)
            return  -1*(env.board.official_score+7.5)



    def minimax(self,parent_env,color,depth,isPass):
        b=parent_env.board
        if(depth==0):
            v=self.getVal(parent_env,False)
            return v,0
        legal_coords = b.get_legal_coords(color)
        fir=True
        for coord in legal_coords:
            action = _coord_to_action(b, coord)
            #print(action,color)
            if(action==self.board_size*self.board_size):
                if(isPass):
                    v = self.getVal(parent_env,True)
                else:
                    v,_ = self.minimax(parent_env.act(action),3-color,depth-1,True)
            else:
                v,_ = self.minimax(parent_env.act(action),3-color,depth-1,False)
            #print(action,v,color)
            if(fir or (v>m if color==1 else v<m) ):
                m=v
                ans=[action]
                fir=False
            elif(v==m):
                ans.append(action)
        ans=np.array(ans)
        r=np.random.choice(ans)
        return m,r