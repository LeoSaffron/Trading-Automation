# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 01:16:15 2020

@author: jonsnow
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
from threading import Timer
import pandas as pd

class Place_Order_Wrapper(EWrapper, EClient):
    ticker_order = ""
    order_side = "BUY"
    order_exchange = "SMART"
    order_currecy = "USD"
    order_primary_exchange = ""
    order_sectype = "STK"
    order_quantity = ""
    order_price = ""
    order_type = "LMT"
    
    order_status = "first"
    order_filled = "first"
    order_remaining = "first"
    order_lastFillPrice = "first"
#    sent_odrders_buy = pd.DataFrame()
#    sent_odrders_sell = pd.DataFrame()
    
    
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId , errorCode, errorString):
        print("Error: ", reqId, " ", errorCode, " ", errorString)

    def nextValidId(self, orderId ):
        self.nextOrderId = orderId
        self.start()

    def orderStatus(self, orderId , status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        print("OrderStatus. Id: ", orderId, ", Status: ", status, ", Filled: ", filled, ", Remaining: ", remaining, ", LastFillPrice: ", lastFillPrice)
        self.order_status = status
        self.order_filled = filled
        self.order_remaining = remaining
        self.order_lastFillPrice = lastFillPrice
        

    def openOrder(self, orderId, contract, order, orderState):
        print("OpenOrder. ID:", orderId, contract.symbol, contract.secType, "@", contract.exchange, ":", order.action, order.orderType, order.totalQuantity, orderState.status)

    def execDetails(self, reqId, contract, execution):
        print("ExecDetails. ", reqId, contract.symbol, contract.secType, contract.currency, execution.execId,
              execution.orderId, execution.shares, execution.lastLiquidity)

#    def start(self):
#        contract = Contract()
#        contract.symbol = "TSLA"
#        contract.secType = "STK"
#        contract.exchange = "SMART"
#        contract.currency = "USD"
#        contract.primaryExchange = "NASDAQ"
#
#        order = Order()
#        order.action = "BUY"
#        order.totalQuantity = 10
#        order.orderType = "LMT"
#        order.lmtPrice = 520

    def start(self):
        contract = Contract()
        contract.symbol = self.ticker_order
        contract.secType = self.order_sectype
        contract.exchange = self.order_exchange
        contract.currency = self.order_currecy
        contract.primaryExchange = self.order_primary_exchange
        

        order = Order()
        order.action = self.order_side
        order.totalQuantity = self.order_quantity
        order.orderType = self.order_type
        if (not self.order_type == "MKT"):
            order.lmtPrice = self.order_price
        order.transmit = True
        
#        self.sent_odrders_buy = pd.DataFrame([] , index = [], columns = ['OrderID', 'Status', 'Filled', "Remaining", "LastFillPrice"])
#        self.sent_odrders_sell = pd.DataFrame([] , index = [], columns = ['OrderID', 'Status', 'Filled', "Remaining", "LastFillPrice"])

        self.placeOrder(self.nextOrderId, contract, order)

    def stop(self):
        self.done = True
        self.disconnect()
        
    def setOrderDetails(self, ticker, primaryExchange, quantity, side, price = 0,
                        order_exchange = "SMART", order_currecy = "USD", order_sectype = "STK", order_type = "LMT"):
        self.ticker_order = ticker
        self.order_primary_exchange = primaryExchange
        self.order_quantity = quantity
        if(price > 0):
            self.order_price = price
        self.order_side = side
        self.order_exchange = order_exchange
        self.order_currecy = order_currecy
        self.order_sectype = order_sectype
        self.order_type = order_type

def place_order(ticker, primaryExchange, quantity, side, price = 0,
                        order_exchange = "SMART", order_currecy = "USD", order_sectype = "STK", order_type = "LMT"):
    app = Place_Order_Wrapper()
    app.setOrderDetails(ticker, primaryExchange, quantity, side, price = price,
                        order_exchange = order_exchange, order_currecy = order_currecy,
                        order_sectype = order_sectype, order_type = order_type)
    app.nextOrderId = 0
    app.connect("127.0.0.1", 7497, 9)

    Timer(3, app.stop).start()
    app.run()
#    return app.nextOrderId
    return app

def main():
    app = Place_Order_Wrapper()
    app.setOrderDetails("VSTO", "IEX", 12, 21, "BUY")
    app.nextOrderId = 0
    app.connect("127.0.0.1", 7497, 9)

    Timer(3, app.stop).start()
    app.run()

if __name__ == "__main__":
    main()