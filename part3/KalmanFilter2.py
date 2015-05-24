import numpy
import pandas
from pykalman import KalmanFilter


def initialize(context):
    # Portfolio
    context.stocks = [sid(8554), sid(8347), sid(23112)]
    # context.stocks = [sid(8554), sid(8347)]
    
    # Parameters for each kalman  filter
    context.params = { stock:{ "init_xhat": 0.0, 
                                "init_P": 1.0, 
                                "Q": 1e-5, 
                                "R": 0.1**2, 
                                "orderSize": 1000, 
                                "percentChange": 0.0015} for stock in context.stocks }
    
    context.stopLoss = False
    
    # Custom params per stock
    context.params[ sid(8554) ]["historicalDays"] = [7]
    context.params[ sid(8347) ]["historicalDays"] = [7]
    # context.params[ sid(8347) ]["percentChange"] = 0.015
    # context.params[ sid(8347) ]["orderSize"] = 10000
    # context.params[ sid(8347) ]["R"] = 0.05**2
    context.params[ sid(23112) ]["historicalDays"] = [7]
    # context.params[ sid(23112) ]["percentChange"] = 0.017
    # context.params[ sid(23112) ]["orderSize"] = 10000
    # context.params[ sid(23112) ]["R"] = 0.05**2

    
    
    # Mapping of stock to its list of filters
    context.models = {} 
    
    # State variables for each kalman filter
    context.order_info = { stock:None for stock in context.stocks }
    context.correct = { stock:0 for stock in context.stocks }
    context.total = { stock:0 for stock in context.stocks }

def handle_data(context, data):
    # For each stock in out portfolio
    for stock in context.stocks:
        
        ###
        ### Handle prediction / order from previous day
        ###
 
        # If there was a previous order ...             
        if context.order_info[stock]:
            # We should cash in and hopefully get some profit!
            order(stock, -context.order_info[stock]["order"].amount)
            
            # Update total number of times we made a predicition
            context.total[stock] = context.total[stock] + 1
            
            # Update number of times we were right about the predicition
            if data[stock].price > context.order_info[stock]["price"] and context.order_info[stock]["predict"] == "up":
                log.info("Right boss!")
                context.correct[stock] = context.correct[stock] + 1
            elif data[stock].price < context.order_info[stock]["price"] and context.order_info[stock]["predict"] == "down":
                log.info("Right boss!")
                context.correct[stock] = context.correct[stock] + 1
            else:
                log.info("Sooooorry!")
            
            # Calculate current accuracy and plot
            accuracy = context.correct[stock] / float(context.total[stock])
            record(accuracy=accuracy, correct=context.correct[stock], total=context.total[stock])
            log.info("Current accuracy for stock %s: %f" % (stock, accuracy))
        


        ###        
        ### Determine what to do this day
        ###
        
        # Grab the maximum number of historical days we need to start the kalman filters
        historical_data = history(bar_count=max(context.params[stock]["historicalDays"]), frequency='1d', field='price')[stock]
        
        # Append today's price
        df = pandas.DataFrame([{0: data[stock].price}])
        historical_data = historical_data.append(df)
        
        # For each model we want to train on this stock
        predictions = []
        for modelSize in context.params[stock]["historicalDays"]:
            # Create a mapping of modelSize to model
            # ie. a kalman filter that has mapSize input, which is declared in historicalDays param
            context.models[stock] = {}

            # Create a kalman filter on modelSize number of data, based on this stock's custom params
            context.models[stock][modelSize] = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
            
            # Process modelSize inputs
            measurements = numpy.asarray(historical_data[0][-modelSize-1:])
            
            # Get tomorrow's predicition
            (filtered_state_means, filtered_state_covariances) = context.models[stock][modelSize].filter(measurements)
            currPrediction = filtered_state_means[-1]
            log.info("currPredicition: %s" % currPrediction)
            log.info("currPrice: %s" % data[stock].price)

            predictions.append(currPrediction)
            #log.info("Prediciton: %f" % currPrediction) 
        
        # How many of the models say we are going to increase or decrease in value?
        votesForUp = sum([ prediction >= (1 + context.params[stock]["percentChange"]) * data[stock].price for prediction in predictions ])
        votesForDown = sum([ prediction <= (1 - context.params[stock]["percentChange"]) * data[stock].price for prediction in predictions ])
        #log.info("Votes for up: %d" % votesForUp) 
        #log.info("Votes for down: %d" % votesForDown) 
        
        # IF we have majority saying Up, then ...
        if votesForUp > len(context.params[stock]["historicalDays"]) / 2:
            # Long the stock since we think it'll go down up value
            log.info("bp: predict up")
            currOrder = order(stock, context.params[stock]["orderSize"])
            # currOrder = order_target_percent(stock, 0.2)
            
            # Stop loss
            if context.stopLoss:
                order(stock, -context.params[stock]["orderSize"], stop_price = 0.95 * data[stock].price)
            
            currOrderObj = get_order(currOrder)
            log.info(currOrderObj)
            context.order_info[stock] = {"predict": "up", 
                                         "price": data[stock].price,
                                        "order": currOrderObj}
        # If we have majority saying Down, then ...
        elif votesForDown > len(context.params[stock]["historicalDays"]) / 2:
            # Short the stock since we think it'll go down in value
            log.info("bp: predict down")
            currOrder = order(stock, -context.params[stock]["orderSize"])
            #currOrder = order_target_percent(stock, -0.2)
            
            # Stop loss
            if context.stopLoss:
                order(stock, context.params[stock]["orderSize"], stop_price = 0.95 * data[stock].price)
            
            currOrderObj = get_order(currOrder)
            log.info(currOrderObj)
            context.order_info[stock] = {"predict": "down", 
                                         "price": data[stock].price,
                                        "order": currOrderObj}
        # Otherwise, don't do a thing (make sure to indicate we did not make an order)
        else:
            context.order_info[stock] = None