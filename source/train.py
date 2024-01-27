from datetime import datetime
import pandas as pd
import qlearn_agent
import importlib
import environment
import argparse
import sys
import os
import tqdm
import logging
import random
import json
sys.path.append(os.environ["PREPROCESS_DIR"])
import preprocess_pkg

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

sys.path.append(os.environ["MODEL_DIR"])


def preprocess_stock(odf, features):
    df = odf.iloc[::-1]
    #df = preprocess_pkg.price_util.min_max_normalize(df)
    for feature, param in features.items():
        if feature == "price":
            df['price'] = df['종가']
        elif feature == "bollinger":
            for i in (range(len(param['n']))):
                df = preprocess_pkg.price_util.add_bollinger_bands(df, param['n'][i], param['m'][i])
        elif feature == "rsi":
            for i in (range(len(param['n']))):
                df = preprocess_pkg.price_util.add_rsi(df, param['n'][i])
        elif feature == "mfi":
            for i in (range(len(param['n']))):
                df = preprocess_pkg.price_util.add_mfi(df, param['n'][i])
    df = df.dropna(how='any')
    res = df.reset_index(drop=True)
    try: df = res.drop(columns=['Unnamed: 0'])
    except: pass
    return df

def run_episode(env, agent, stock_dir):
    for ei in range(agent.epoch):
        total_reward = 0
        total_profit = 0
        eoutf = open(os.path.join(stock_dir, "epoch_%d.jsonl" % (ei)), "w", encoding="utf8")
        di = 0
        while not env.done:
            # agent takes action & interact with env
            today = env.stock_state['날짜']
            price = env.stock_state['종가']
            stock_state = env.stock_df.iloc[env.idx]
            model_state = agent.model.preprocess(env, agent)
            env.model_state = model_state
            typ, action, reward, profit, qvals, pchange = agent.act(env)
            #print(price, action, reward)
            env.step()
            if not env.done:
                model_next_state = agent.model.preprocess(env, agent)
                env.model_next_state = model_next_state
            else: env.model_next_state = None
            total_profit += profit
            total_reward += reward
            # update agent memory
            # ["stock_state", "state", "action", "next_state", "reward", "done"]
            mem = {'stock_state' : stock_state,
                    'state': model_state,
                   'price' :  [price, env.stock_state['종가']],
                    'action': action,
                    'next_state': model_next_state,
                    'reward': reward,
                    'done': env.done}
            agent.update_memory(list(mem.values()))

            # update model
            loss = agent.update_agent_nn()
            # write history
            di += 1
            ologs = {"date" : di,
                     "price" : price,
                     "action" : "%s_%s" % (typ, "enter" if action == 1 else "exit"),
                     'pchange': pchange,
                     "reward" : reward,
                     "state" : [[x.tolist() for x in model_state], agent.model.feature_names],
                     "inven" : agent.inventory,
                     "qvals" : qvals,
                     "total_reward" : total_reward,
                     "total_profit" : total_profit}
            eoutf.write(json.dumps(ologs, ensure_ascii=False) + "\n")
            # write train progress
            outfs['train_log'].write("\t".join([str(env.stock_name) + "_%s" % (str(ei)),
                                                str(loss),
                                                str(total_reward)]) + "\n")

        eoutf.close()
        # write train progress
        agent.reset()
        env.reset()


if __name__ == "__main__":
    global agent
    global env
    global outfs

    ctime = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser(
        prog='RL_train',
        description='Trains a reinforcement learning model for stock trading',
        epilog='Woo June Cha (wjcha96@gmail.com), Eunji Jeong()')
    parser.add_argument('--name',
                        default='%s' % (ctime),
                        help='name id for train run')
    parser.add_argument('--model',
                        default='linear_regression',
                        help='pytorch NN model for rl learning')
    parser.add_argument('--stock_dir',
                        default='/Users/user/projects/DLBollingerTrader_dev/scraping/data/20230818/prices/',
                        #default='/Users/user/projects/DLBollingerTrader_dev/trade_rl/data/train_dummy/',
                        help='input stock data')
    args = parser.parse_args()
    model_lib = importlib.import_module(args.model)
    model = model_lib.Model()
    logging.info("model loaded")
    agent = qlearn_agent.Agent(model)
    logging.info("agent generated")

    # set train path
    train_in_dir = """%s""" % (args.stock_dir)
    train_out_dir = """%sdata/train_out/%s/""" % (os.environ["RL_DIR"], args.name)
    episode_dir = os.path.join(train_out_dir, "episodes")
    outfs = {"train_log" : open(os.path.join("""%s%s""" % (train_out_dir, "train_log.txt")), "w", encoding="utf8")}

    # retrieve train stock info
    stock_fs = os.listdir(train_in_dir)
    stock_info = {}
    for stock_f in stock_fs[:10]:
        stock_df = pd.read_csv("""%s%s""" % (train_in_dir, stock_f), delimiter='\t')
        stock_info[stock_f] = {"stock_df" : stock_df}

    # train start
    stock_num = 0
    episode_num = 0
    for stock_f, info in tqdm.tqdm(stock_info.items(), desc="training stocks"):
        stock_df = preprocess_stock(info['stock_df'], agent.model.features)
        indices = [x * agent.model.window * 2 for x in range(len(stock_df) // (agent.model.window*2))]
        indices = ([(indices[i], indices[i + 1]) for i in range(len(indices) - 1)])
        random.shuffle(indices)
        #print("indices : ", len(indices))
        for i, index in tqdm.tqdm(enumerate(indices), desc="episodes"):
            s, e = index
            episode_num += 1
            stock_name = "%d_%s_episode_%d" % ((stock_num), stock_f.replace(".tsv",""), i)
            stock_dir = os.path.join(train_out_dir, "episodes", stock_name)
            os.mkdir(stock_dir)
            cstock = stock_df[s:e].reset_index()
            env = environment.Env(cstock, stock_name, agent.model.window)
            logging.info("environment created for (%s)" % (stock_name))
            run_episode(env, agent, stock_dir)
            episode_num += 1
        stock_num += 1

    # end
    for outf in outfs.values(): outf.close()


