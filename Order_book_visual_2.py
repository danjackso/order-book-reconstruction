import pandas as pd
import numpy as np
from datetime import datetime
import ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re

"""
data = pd.read_csv('test1.txt', nrows=10000, usecols=[0, 1, 2, 3, 4, 5, 6, 7], index_col='# Date')
data.fillna(0, inplace=True)
data = data.astype(
    {'type': 'int8', 'side': 'int8', 'reason': 'int8', 'price': 'float32', 'order_id': 'float32', 'size': 'float32',
     'remaining_size': 'float32'})
print(data)
data.index = pd.to_datetime(data.index, unit='s')

print(data.info(memory_usage=True))
"""

data = pd.read_csv('coinbase_btc-usd.csv', nrows=100000, delimiter=';', usecols=[1])

class Order_Book():
    plt.axis([-0.01, 0, 0, 1000])

    def __init__(self):
        self.latest_time=0

        self.received_sell_quotes_limit = np.zeros((10000, 4))
        self.open_sell_quotes_limit = np.zeros((10000, 5))

        self.received_buy_quotes_limit = np.zeros((10000, 4))
        self.open_buy_quotes_limit = np.zeros((10000, 5))

    def received_buy_quotes(self, quote):
        idx = np.where(self.received_buy_quotes_limit[:, 0] == 0)[0][0]
        if quote['order_type'] == 'limit':
            self.received_buy_quotes_limit[idx, 0] = datetime.strptime(quote['time'],
                                                                       "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            self.received_buy_quotes_limit[idx, 1] = re.sub('\D', '', quote['order_id'])
            self.received_buy_quotes_limit[idx, 2] = quote['size']
            self.received_buy_quotes_limit[idx, 3] = quote['price']

    def received_sell_quotes(self, quote):
        idx = np.where(self.received_sell_quotes_limit[:, 0] == 0)[0][0]
        if quote['order_type'] == 'limit':
            self.received_sell_quotes_limit[idx, 0] = datetime.strptime(quote['time'],
                                                                        "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            self.received_sell_quotes_limit[idx, 1] = re.sub('\D', '', quote['order_id'])
            self.received_sell_quotes_limit[idx, 2] = quote['size']
            self.received_sell_quotes_limit[idx, 3] = quote['price']

    def open_buy_quotes(self, quote):
        idx = np.where(self.open_buy_quotes_limit[:, 0] == 0)[0][0]
        if idx == 0:
            self.open_buy_quotes_limit[idx, 0] = datetime.strptime(quote['time'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            self.open_buy_quotes_limit[idx, 1] = int(re.sub('\D', '', quote['order_id']))
            self.open_buy_quotes_limit[idx, 2] = float(quote['remaining_size'])
            self.open_buy_quotes_limit[idx, 3] = float(quote['price'])

        if float(quote['price']) >= 0.995 * self.open_buy_quotes_limit[0, 3]:
            self.open_buy_quotes_limit[idx, 0] = datetime.strptime(quote['time'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            self.open_buy_quotes_limit[idx, 1] = int(re.sub('\D', '', quote['order_id']))
            self.open_buy_quotes_limit[idx, 2] = float(quote['remaining_size'])
            self.open_buy_quotes_limit[idx, 3] = float(quote['price'])

            self.open_buy_quotes_limit[:idx + 1, :] = self.open_buy_quotes_limit[
                np.argsort(self.open_buy_quotes_limit[:idx + 1, 3])[::-1]]

            unique_values = np.unique(self.open_buy_quotes_limit[:idx + 1, 3])

            self.open_buy_quotes_limit[:, 4]=0
            for item in unique_values:
                where = np.where(self.open_buy_quotes_limit[:, 3] == item)[0]
                self.open_buy_quotes_limit[where, 4] = np.sum(self.open_buy_quotes_limit[:where[-1] + 1, 2])

        idx = np.where(self.received_buy_quotes_limit[:, 1] == self.open_buy_quotes_limit[idx, 1])
        self.received_buy_quotes_limit[idx, :] = 0

    def open_sell_quotes(self, quote):
        idx = np.where(self.open_sell_quotes_limit[:, 0] == 0)[0][0]

        if idx == 0:
            self.open_sell_quotes_limit[idx, 0] = datetime.strptime(quote['time'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            self.open_sell_quotes_limit[idx, 1] = int(re.sub('\D', '', quote['order_id']))
            self.open_sell_quotes_limit[idx, 2] = float(quote['remaining_size'])
            self.open_sell_quotes_limit[idx, 3] = float(quote['price'])

        if float(quote['price']) <= 1.005 * self.open_sell_quotes_limit[0, 3]:
            self.open_sell_quotes_limit[idx, 0] = datetime.strptime(quote['time'], "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            self.open_sell_quotes_limit[idx, 1] = int(re.sub('\D', '', quote['order_id']))
            self.open_sell_quotes_limit[idx, 2] = float(quote['remaining_size'])
            self.open_sell_quotes_limit[idx, 3] = float(quote['price'])

            self.open_sell_quotes_limit[:idx + 1, :] = self.open_sell_quotes_limit[
                np.argsort(self.open_sell_quotes_limit[:idx + 1, 3])]

            unique_values = np.unique(self.open_sell_quotes_limit[:idx + 1, 3])
            self.open_sell_quotes_limit[:, 4]=0
            for item in unique_values:
                where = np.where(self.open_sell_quotes_limit[:, 3] == item)[0]
                self.open_sell_quotes_limit[where, 4] = np.sum(self.open_sell_quotes_limit[:where[-1] + 1, 2])

        idx = np.where(self.received_sell_quotes_limit[:, 1] == self.open_sell_quotes_limit[idx, 1])
        self.received_sell_quotes_limit[idx, :] = 0

    def done_buy_quotes(self, quote):
        idx = np.where(self.open_buy_quotes_limit[:, 1] == int(re.sub('\D', '', quote['order_id'])))
        if len(idx[0]) > 0:
            self.open_buy_quotes_limit[idx, :] = 0
            self.open_buy_quotes_limit = self.open_buy_quotes_limit[np.argsort(self.open_buy_quotes_limit[:, 3])[::-1]]

    def done_sell_quotes(self, quote):
        idx = np.where(self.open_sell_quotes_limit[:, 1] == int(re.sub('\D', '', quote['order_id'])))
        if len(idx[0]) > 0:
            self.open_sell_quotes_limit[idx, :] = 0
            self.open_sell_quotes_limit = self.open_sell_quotes_limit[
                np.argsort(self.open_sell_quotes_limit[:, 3])[::-1]]
            idx = np.where(self.open_sell_quotes_limit[:, 0] == 0)[0][0]
            self.open_sell_quotes_limit[:idx, :] = self.open_sell_quotes_limit[
                np.argsort(self.open_sell_quotes_limit[:idx, 3])]

    def change_buy_quotes(self, quote):
        idx = np.where(self.open_buy_quotes_limit[:, 1] == int(re.sub('\D', '', quote['order_id'])))
        self.open_buy_quotes_limit[idx, 2] = quote['new_size']

    def change_sell_quotes(self, quote):
        idx = np.where(self.open_sell_quotes_limit[:, 1] == int(re.sub('\D', '', quote['order_id'])))
        self.open_sell_quotes_limit[idx, 2] = quote['new_size']

    def plot_quotes(self):
        plt.cla()
        midprice=(self.open_buy_quotes_limit[0, 3]+self.open_sell_quotes_limit[0, 3])/2

        idx = np.where(self.open_buy_quotes_limit[:, 0] == 0)[0][0]
        X_data = np.unique(self.open_buy_quotes_limit[:idx, 3] / midprice - 1)[::-1]
        y_data = np.unique(self.open_buy_quotes_limit[:idx, 4])

        X = np.linspace(0, -0.005, 50)
        y = np.empty((50, 1)).ravel()

        for index_bins, item in enumerate(X):
            where = np.where(X_data >= item)[0]
            y[index_bins] = y_data[where[-1]] if len(where)>0 else 0
        X[0]=X[1]

        x=np.zeros((50,1)).ravel()
        plt.fill_between(X,x,y,color='g',alpha=0.5)


        idx = np.where(self.open_sell_quotes_limit[:, 0] == 0)[0][0]
        X_data = np.unique(self.open_sell_quotes_limit[:idx, 3] / midprice - 1)
        y_data = np.unique(self.open_sell_quotes_limit[:idx, 4])

        X = np.linspace(0, 0.005, 50)
        for index_bins, item in enumerate(X):
            where = np.where(X_data <= item)[0]
            y[index_bins] = y_data[where[-1]] if len(where)>0 else 0

        X[0]=X[1]
        plt.fill_between(X,x,y,color='r',alpha=0.5)
        plt.text(0,1400,'time = {}'.format(self.latest_time),horizontalalignment='center')
        plt.text(0,1300,'Spread = {:.4f}'.format(self.open_buy_quotes_limit[0, 3]-self.open_sell_quotes_limit[0, 3]),horizontalalignment='center')
        plt.title('Bitcoin//USD order book')
        plt.xlim([-0.005,0.005])
        plt.ylim([0,1500])

order_book = Order_Book()
start = datetime.strptime(ast.literal_eval(data.values[0][0])['time'], "%Y-%m-%dT%H:%M:%S.%fZ")
end = datetime.strptime(ast.literal_eval(data.values[-1][0])['time'], "%Y-%m-%dT%H:%M:%S.%fZ")

for index, incoming_quote in enumerate(range(0, data.shape[0])):
    incoming_quote = ast.literal_eval(data.values[incoming_quote][0])
    order_book.latest_time=datetime.strptime(incoming_quote['time'], "%Y-%m-%dT%H:%M:%S.%fZ")

    if incoming_quote['type'] == 'received' and incoming_quote['side'] == 'buy':
        order_book.received_buy_quotes(incoming_quote)

    elif incoming_quote['type'] == 'open' and incoming_quote['side'] == 'buy':
        order_book.open_buy_quotes(incoming_quote)

    elif incoming_quote['type'] == 'done' and incoming_quote['side'] == 'buy':
        order_book.done_buy_quotes(incoming_quote)

    elif incoming_quote['type'] == 'change' and incoming_quote['side'] == 'buy':
        order_book.change_buy_quotes(incoming_quote)

    elif incoming_quote['type'] == 'received' and incoming_quote['side'] == 'sell':
        order_book.received_sell_quotes(incoming_quote)

    elif incoming_quote['type'] == 'open' and incoming_quote['side'] == 'sell':
        order_book.open_sell_quotes(incoming_quote)

    elif incoming_quote['type'] == 'done' and incoming_quote['side'] == 'sell':
        order_book.done_sell_quotes(incoming_quote)

    elif incoming_quote['type'] == 'change' and incoming_quote['side'] == 'sell':
        order_book.change_sell_quotes(incoming_quote)

    if index%100==0 and index>=10000:
        order_book.plot_quotes()
        plt.savefig("value{}.png".format(index))
        plt.pause(1e-11)
        if index==12000:
            break

plt.show()

'''
Try and eliminate quotes outside of the 10% range

spoofing quotes
'''
