# order-book-reconstruction

Reconstructed trading order book using json objects of trade events. 
Downloaded bitcoin//usd trade data from coinbase in csv format where read into a pandas dataframe in chunks of 100,000 rows for efficent memory management.

Python code chronologically processes each json object and plots the limit order quote on the graph below. This graph details the depth of market against percentage distance from best bid/ask price, current time and the current spread.

![Test video 1](Order_book_video.gif)
# Example of Json Trade Quotes
{'type': 'received', 'side': 'buy', 'product_id': 'BTC-USD', 'time': '2020-06-15T00:00:06.759655Z',\
'order_id': 'fddbd1ac-caa4-4448-ab89-6162aad85381', 'order_type': 'limit', 'size': '0.639', 'price': '9325.05'}

{'type': 'open', 'side': 'buy', 'product_id': 'BTC-USD', 'time': '2020-06-15T00:00:05.339048Z','price': '9328.08', 'order_id': 'dc6fa905-e03a-4ed5-a9a7-4c296e788011', 'remaining_size': '0.01'}

{'type': 'done', 'side': 'buy', 'product_id': 'BTC-USD', 'time': '2020-06-15T00:00:07.050407Z', 'order_id': '45a264d9-8077-4b2d-893d-91d39555bdf6', 'reason': 'filled', 'price': '9328.7', 'remaining_size': '0'}


# Future Work
1. To develop custom search function to speed up dataprocessing to be able to process quotes in realtime.
2. Speed up plotting by using matplotlib.animate
3. Use CPU multiprocessing and/or GPU in code by using one process for each buy quotes and seel quotes.
4. Intergrate orderbook into deep high frequency trading reinforcement learning trading bot.

## Requirements
* python 3.5.x
* pandas 1.1
* numpy 1.15.0
* matplotlib 3.3.0
* sys
* re
* ast
* datetime
