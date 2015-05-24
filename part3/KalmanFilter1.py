import numpy
import pandas

class KalmanFilter(object):
    """ Class that implements a kalman filter. Based off of http://wiki.scipy.org/Cookbook/KalmanFiltering. """

    def __init__(self, size, init_xhat=0.0, init_P = 1.0, Q=1e-5, R=0.1**2):
        """ Init the kalman filter. Only needs to know the size of the input measurements. All other parameters
        are set to the defaults in the reference implementation. """
        self.size = size

        self.Q = Q                              # Process variance
        self.R = R                              # estimate of Measurement variance, change to see effect

        # Pre-allocate space for arrays
        self.xhat = numpy.zeros(self.size)      # Aposteri estimate of x
        self.P = numpy.zeros(self.size)         # Aposteri error estimate
        self.xhatminus = numpy.zeros(self.size) # Apriori estimate of x
        self.Pminus = numpy.zeros(self.size)    # Apriori error estimate
        self.K = numpy.zeros(self.size)         # Kalman factor

        # intial guesses
        self.xhat[0] = init_xhat
        self.P[0] = init_P

    def processInput(self, z):
        """ z - input measurements, must be of length self.size """

        # For each input measurement, do a process and update step
        for k in range(1, self.size):
            # time update
            self.xhatminus[k] = self.xhat[k-1]
            self.Pminus[k] = self.P[k-1] + self.Q

            # measurement update
            self.K[k] = self.Pminus[k] / ( self.Pminus[k] + self.R )
            self.xhat[k] = self.xhatminus[k] + self.K[k] * (z[k] - self.xhatminus[k])
            self.P[k] = (1 - self.K[k]) * self.Pminus[k]

    def predict(self):
        """ Return whatever the kalman filter is prediciting right now, which is whatever the last element of xhat is """
        return self.xhat[-1]


def initialize(context):
    # Portfolio
    context.stocks = [sid(8554), sid(8347), sid(23112)]
    # context.stocks = [sid(7784)]
    
    # Parameters for each kalman  filter
    context.params = { stock:{ "init_xhat": 0.0, 
                                "init_P": 1.0, 
                                "Q": 1e-5, 
                                "R": 0.1**2, 
                                "orderSize": 5000, 
                                "percentChange": 0.02} for stock in context.stocks }
    
    # Custom params per stock
    context.params[ sid(8554) ]["historicalDays"] = [7, 15, 30]
    context.params[ sid(8347) ]["historicalDays"] = [7, 10]
    context.params[ sid(8347) ]["percentChange"] = 0.015
    context.params[ sid(8347) ]["orderSize"] = 10000
    context.params[ sid(8347) ]["R"] = 0.05**2
    context.params[ sid(23112) ]["historicalDays"] = [7]
    context.params[ sid(23112) ]["percentChange"] = 0.017
    context.params[ sid(23112) ]["orderSize"] = 10000
    context.params[ sid(23112) ]["R"] = 0.05**2

    
    
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
        #
        # If there was a previous order ...             
        if context.order_info[stock]:
            # We should cash in and hopefully get some profit!
            order(stock, -context.order_info[stock]["orderSize"].amount)
            
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
            context.models[stock][modelSize] = KalmanFilter(size=modelSize + 1, 
                                                        init_xhat=context.params[stock]["init_xhat"], 
                                                        init_P=context.params[stock]["init_P"],
                                                        Q=context.params[stock]["Q"],
                                                        R=context.params[stock]["R"])

            # Process modelSize inputs
            context.models[stock][modelSize].processInput(historical_data[0][-modelSize-1:])
            
            # Get tomorrow's predicition
            currPrediction = context.models[stock][modelSize].predict()
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
            # order(stock, context.params[stock]["orderSize"])
            currOrder = order_percent(stock, 0.5)
            currOrderObj = get_order(currOrder)
            log.info(currOrderObj)
            context.order_info[stock] = {"predict": "up", 
                                         "price": data[stock].price,
                                        "orderSize": currOrderObj}
        # If we have majority saying Down, then ...
        elif votesForDown > len(context.params[stock]["historicalDays"]) / 2:
            # Short the stock since we think it'll go down in value
            log.info("bp: predict down")
            # order(stock, -context.params[stock]["orderSize"])
            currOrder = order_percent(stock, -0.5)
            currOrderObj = get_order(currOrder)
            log.info(currOrderObj)
            context.order_info[stock] = {"predict": "down", 
                                         "price": data[stock].price,
                                        "orderSize": currOrderObj}
        # Otherwise, don't do a thing (make sure to indicate we did not make an order)
        else:
            context.order_info[stock] = None