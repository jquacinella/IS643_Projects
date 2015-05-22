import datetime as dt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import numpy as np


def test_coint(pair):
    # Using Augmented Dickey-Fuller unit root test (from Brett's code)
    #result = sm.OLS(pair[1], pair[0]).fit()
    #dfResult =  ts.adfuller(result.resid)
    #return dfResult[0] >= dfResult[4]['10%']

    # Using cointegration tets built into statsmodels
    result = ts.coint(pair[1], pair[0])
    return result[0] >= result[2][2]

def initialize(context):
    # Initialize stock universe with the following stocks:  
    context.stocks = [
                    # (sid(4283), sid(5885)),   # Coke and Pepsi
                    # (sid(8229), sid(21090)),  # Walmart and Target
                    # (sid(8347), sid(23112)),  # Exxon mobile and Chevron
                    # (sid(3496), sid(4521)),   # Home depot and lowes
                    # (sid(20088), sid(25006)), # Goldman Sachs, JP Morgan
                    # (sid(863), sid(25165)),   # BHP Billiton Limited (BHP) and BHP Billiton plc (BBL)
                    # (sid(1638), sid(1637)),   # Comcact K and Comcast A
                    # (sid(7784), sid(7767)),   # Unilever
                    # (sid(8554), sid(2174))    # SPY and DIA 
                ]


    
    context.spreads = []  
    context.capital_base = 100000
    context.posSizeX = context.capital_base  
    context.posSizeY = context.capital_base           #I assumed 50% margin for both long and short trades  
    
    context.coint_window_length = 60
    
    context.params = dict((pair, {"thresholdEnter": 2.0, "thresholdExit": 2.0, "window_length": 14, "stopLossOrder": False}) for pair in context.stocks)
    
    #context.params[(sid(863), sid(25165))]["window_length"] = 14
    #context.params[(sid(863), sid(25165))]["thresholdExit"] = 2.0
    #context.params[(sid(863), sid(25165))]["stopLossOrder"] = True
    #context.params[(sid(1638), sid(1637))]["stopLossOrder"] = True

    context.wasCointegrated = dict((pair, False) for pair in context.stocks)
    context.cointegrated = dict((pair, False) for pair in context.stocks)
    context.invested = dict((pair, 0) for pair in context.stocks)
    

def handle_data(context, data):
    # Grab historical data on all stocks
    historical_data = history(context.coint_window_length, '1d', 'price')

    # Loop over all stocks in our portfolio
    for pair in context.stocks:
        # Keep track of the current pair
        (context.currX, context.currY) = pair

        # Keep track of previous cointegrated state
        context.wasCointegrated[pair] = context.cointegrated
        
        window_length = context.params[pair]["window_length"]
        xTimeseries = historical_data[context.currX][-window_length:]
        yTimeseries = historical_data[context.currY][-window_length:]

        # DEtermine if the pair is cointegrated at this point
        context.cointegrated[pair] = test_coint(pair=(xTimeseries, yTimeseries))
        
        # Check if the pair is still cointegrated ...    
        if not context.cointegrated[pair]:
            log.info("Not cointegrated!")
            
            # ... and if not, sell our position ASAP
            if context.wasCointegrated[pair]:
                sell_spread(context)
            
            # Go to the next pair
            continue
        
        # The current price values, their ratio and their spread?
        currXPrice = data[context.currX].price
        currYPrice = data[context.currY].price
        currRatio = currXPrice / currYPrice
        currSpread = currXPrice - currYPrice
        record(currSpread=currSpread)

        # Historical spread from last window_size days, as well as the historical spread mean and sd
        spread = xTimeseries - yTimeseries
        spreadMean = np.mean(spread)
        spreadSD = np.std(spread)
        record(spreadMeanPlus=spreadMean+(2.0*spreadSD))
        record(spreadMeanMinus=spreadMean-(2.0*spreadSD))


        # Given the above, check if we should place an order
        place_orders(context, data, currSpread, spreadMean, spreadSD)


# def compute_zscore(context, data):  
#     #spread = data[context.currX].price / data[context.currY].price
#     spread = data[context.currX].price - data[context.currY].price  
#     context.spreads.append(spread)  
#     spread_wind = context.spreads[-context.window_length:]  
#     zscore = (spread - np.mean(spread_wind)) / np.std(spread_wind)  
#     return zscore



def place_orders(context, data, currSpread, spreadMean, spreadSD):  
    """ Buy spread if zscore is <= -2, sell if zscore >= 2, close the trade when zscore crosses 0 """  
    #if zscore >= context.zThreshold and context.invested == 0:
    pair = (context.currX, context.currY)
    invested = context.invested[pair]
    enterThreshold = context.params[pair]['thresholdEnter']
    exitThreshold = context.params[pair]['thresholdExit']
    
    if currSpread >= spreadMean + enterThreshold * spreadSD and invested == 0:  
        log.info("Condition 1: Shorting %s, Longing %s" % (context.currX, context.currY))

        context.params[pair].update({"transactionMean": spreadMean, "transactionSD": spreadSD})

        #order(context.currX, -9000)  
        #order(context.currY, 9000) 
        order(context.currX, -int(context.posSizeX / data[context.currX].price))  
        order(context.currY, int(context.posSizeY / data[context.currY].price)) 

        if context.params[pair]["stopLossOrder"]:
            order(context.currY, int(context.posSizeX / data[context.currX].price), stop_price = data[context.currX].price * 0.95) 
            order(context.currY, -int(context.posSizeY / data[context.currY].price), stop_price = data[context.currY].price * 0.95) 

        context.invested[pair] = 1  
    #elif zscore <= -context.zThreshold and context.invested == 0:
    elif currSpread <= spreadMean - enterThreshold * spreadSD and invested == 0:
        log.info("Condition 2: Shorting %s, Longing %s" % (context.currY, context.currX))

        context.params[pair].update({"transactionMean": spreadMean, "transactionSD": spreadSD})

        #order(context.currX, 9000)  
        #order(context.currY, -9000)
        order(context.currX, int(context.posSizeX / data[context.currX].price))  
        order(context.currY, -int(context.posSizeY / data[context.currY].price)) 

        if context.params[pair]["stopLossOrder"]:
            order(context.currY, int(context.posSizeY / data[context.currY].price), stop_price = data[context.currY].price * 0.95) 
            order(context.currX, -int(context.posSizeX / data[context.currX].price), stop_price = data[context.currX].price * 0.95) 

        context.invested[pair] = 2  
    #elif (zscore < 1.0 * context.zThreshold and context.invested==1) or (zscore > 1.0 * context.zThreshold and context.invested==2):  
    #elif (invested == 1 and currSpread <= spreadMean + exitThreshold * spreadSD) or \
    #     (invested == 2 and currSpread >= spreadMean - exitThreshold * spreadSD):
    elif (invested == 1 and currSpread <= context.params[pair]["transactionMean"] + exitThreshold * context.params[pair]["transactionSD"]) or \
         (invested == 2 and currSpread >= context.params[pair]["transactionMean"] - exitThreshold * context.params[pair]["transactionSD"]):
        log.info("Selling spread!")
        sell_spread(context)  
        context.invested[pair] = 0 
        log.info('Selling spread')  


def sell_spread(context):  
    """  
    decrease exposure, regardless of posstockB_amountition long/short.  
    buy for a short position, sell for a long.  
    """  
    stockB_amount = context.portfolio.positions[context.currY].amount  
    order(context.currY, -1 * stockB_amount)  
    
    stockA_amount = context.portfolio.positions[context.currX].amount  
    order(context.currX, -1 * stockA_amount)