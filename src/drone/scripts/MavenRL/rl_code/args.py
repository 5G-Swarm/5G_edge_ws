

def generate_args():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-d', dest='description', default='td3_gnn', help='[Method] description.')


    argparser.add_argument('-e', '--env', default='DecFormationFlying-v0', type=str, help='[Env] Available envs, including \
        LunarLander-v2 \
        LunarLanderContinuous-v2 \
        HalfCheetah-v2 \
        Hopper-v2 \
    .')
    argparser.add_argument('-m', '--method', default='None', type=str, help='[Method] Method to use.')

    argparser.add_argument('--model-dir', default='None', type=str, help='[Model] dir contains model (default: False)')
    argparser.add_argument('--model-num', default=-1, type=str, help='[Model] model-num to use.')

    argparser.add_argument('--seed', default=0, type=int, help='seed.')
    argparser.add_argument('--render', action='store_true', help='render the env (default: False)')


    args = argparser.parse_args()
    return args





class EnvParams(object):
    def __init__(self, env):
        self.env_name = str(env).split('<')[-1][:-4]
        
        if self.env_name == 'LunarLander-v2':
            self.dim_state = env.observation_space.shape[0]
            self.dim_action = env.action_space.n
            self.solved_reward = 230
            self.epoch_length = 300
            
        elif self.env_name == 'LunarLanderContinuous-v2':
            self.dim_state = env.observation_space.shape[0]
            self.dim_action = env.action_space.shape[0]
            self.solved_reward = 230
            self.epoch_length = 300

        elif self.env_name == 'HalfCheetah-v2':
            self.dim_state = env.observation_space.shape[0]
            self.dim_action = env.action_space.shape[0]
            self.solved_reward = 500
            self.epoch_length = int(3e10)
        elif self.env_name == 'Hopper-v2':
            self.dim_state = env.observation_space.shape[0]
            self.dim_action = env.action_space.shape[0]
            self.solved_reward = 1000
            self.epoch_length = int(3e10)
        elif self.env_name == "FormationFlying-v2":
            self.dim_state = env.observation_space.shape
            self.dim_action = env.action_space.shape
            print("dim_state",self.dim_state,self.dim_action)
            self.solved_reward = 10000
            self.epoch_length = int(40000)
        elif self.env_name == "SingleFormationFlying-v0":
            self.dim_state = env.observation_space.shape[1]
            self.dim_action = env.action_space.shape[1]
            # import pdb;pdb.set_trace()
            print("dim_state",self.dim_state,self.dim_action)
            self.solved_reward = 10000
            self.epoch_length = int(40000)
        
        elif self.env_name == "DecentralFormationFlying-v0":
            self.dim_state = env.observation_space.shape[1]
            self.dim_action = env.action_space.shape[1]
            # import pdb;pdb.set_trace()
            print("dim_state",self.dim_state,self.dim_action)
            self.epoch_length = int(40000)
        elif self.env_name == "Two_Order_DecentralFormationFlying-v0":
            self.dim_state = env.observation_space.shape[1]
            self.dim_action = env.action_space.shape[1]
            
            print("dim_state",self.dim_state,self.dim_action)
            self.epoch_length = int(40000)
        elif self.env_name == "formation_containment-v0":#bullet_drone
            self.dim_state = 12#env.observation_space.shape[1]
            self.dim_action = 2#env.action_space.shape[1]
            # import pdb;pdb.set_trace()
            # print("dim_state",self.dim_state,self.dim_action)
            self.epoch_length = int(40000)
        elif self.env_name == "formation_containment_2order-v0":#bullet_drone
            self.dim_state = 24#env.observation_space.shape[1]
            self.dim_action = 2#env.action_space.shape[1]
            # import pdb;pdb.set_trace()
            # print("dim_state",self.dim_state,self.dim_action)
            self.epoch_length = int(40000)
        else:
            raise NotImplementedError
        return

