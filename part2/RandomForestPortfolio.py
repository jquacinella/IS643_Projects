from sklearn.ensemble import RandomForestClassifier
from numpy import std, mean
from datetime import timedelta

def initialize(context):    
    # Portfolio
    context.stocks = [sid(8554), sid(4283), sid(5885),
                    # sid(8229), sid(21090), 
                    # sid(8347), sid(23112),
                    # sid(3496), sid(4521),   
                    # sid(20088), sid(25006), 
                    # sid(863), sid(25165),   
                    # sid(1638), sid(1637),   
                    # sid(7784), sid(7767),   
                    # sid(8554), sid(2174)
              ]

    # For debugging purposes, its nice to see output from classifier
    # ISSUE: can only do 5 in quantopinan as of right now
    context.recordKeys = { stock:"prediction%s" % stock for stock in context.stocks[:5] }

    # Params
    context.params = dict((stock, { "years": 5,             # Number of years of data to train on
                                    "historicalDays": 30,   # Number of days to look at in the past when coming up with training data (i.e. input size to the classifier)
                                    "predictionDays": 5,    # Number of days into the future we want to attempt to predict
                                    "percentChange": .02,   # How much does the value need to change to consider it a positive or negative training example?
                                    "orderSize": 2000       # How much to order when we have some prediction we want to bet on
                                }) for stock in context.stocks)

    # Custom params
    # # context.params[sid(8554)]["years"] = 1
    context.params[sid(8554)]["historicalDays"] = 5
    context.params[sid(8554)]["percentChange"] = .015
    
    context.params[sid(4283)]["historicalDays"] = 10
    context.params[sid(4283)]["percentChange"] = .02
    
    context.params[sid(5885)]["historicalDays"] = 10
    context.params[sid(5885)]["percentChange"] = .02
    
    context.params[sid(4521)]["orderSize"] = 5000
    context.params[sid(4521)]["percentChange"] = .015
    
    # context.params[sid(21090)]["historicalDays"] = 60
    # context.params[sid(21090)]["percentChange"] = .02
    
    # State per stock
    context.state = { stock:{"warmup": True, "transactions": {}, "model": None} for stock in context.stocks }


def handle_data(context, data):
    # For each stock
    for stock in context.stocks:
        # Handle any transactions we should sell off to make some moola
        transactionStart = data[stock].datetime - timedelta(days=context.params[stock]["predictionDays"])
        for ts in context.state[stock]["transactions"]:
            if not context.state[stock]["transactions"][ts]["done"] and ts <= transactionStart:
                log.info("bp: cleanup")
                order(stock, -context.state[stock]["transactions"][ts]["size"])
                context.state[stock]["transactions"][ts]["done"] = True

        
        # Train or retrain model on historical data
        # For now, we do this once at the beginning but this should be re-done periodically
        if context.state[stock]["warmup"] or  ("warmupedLast" in context.state[stock] and context.state[stock]["warmupedLast"] + timedelta(days=356) < data[stock].datetime):
            # Create model form historical data
            context.state[stock]["model"] = generateModel(context, stock)

            # Update state
            context.state[stock]["warmup"] = False
            context.state[stock]["warmupedLast"] = data[stock].datetime  

        
        # Calculate input to the RandomForestClassifier, which is a vector of length context.historicalDays
        # made up of reverse percent change
        priceChanges = generatePercentChanges(history(bar_count=context.params[stock]["historicalDays"], frequency='1d', field='price')[stock])
        #log.info("Price Changes: %s" % priceChanges)

        # Pass this to the model and see what the model thinks is going to happen in context.predictionDays
        prediction = context.state[stock]["model"].predict(priceChanges)[0]
        
        # Output to quantopian
        if stock in context.recordKeys:
            record(**{context.recordKeys[stock]: prediction})
    
        # Trade based on the model output and keep track of it (so we can sell later to lock in profit)
        if prediction == 1:
            log.info("bp: predict up")
            order(stock, context.params[stock]["orderSize"])
            context.state[stock]["transactions"][data[stock].datetime] = {"size": context.params[stock]["orderSize"], "done": False}
        elif prediction == -1:
            log.info("bp: predict down")
            # Short the stock since we think it'll go down in value
            order(stock, -context.params[stock]["orderSize"])
            context.state[stock]["transactions"][data[stock].datetime] = {"size": -context.params[stock]["orderSize"], "done": False}





def generateModel(context, stock):
    ''' Function oversees how ot generate a RFC model for a given stock. '''
    # Get the historical data for this stock
    historical_data = history(bar_count=context.params[stock]["years"] * 250, frequency='1d', field='price')[stock]

    # Calculate reverse price changes vector from price column (input) and
    #   calculate the actual movement context.predictionDays later (output)
    (trainingX, trainingY) = generateModelData(context, stock, historical_data)

    # Train the classifier on our training set
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(trainingX, trainingY)
    return(clf)



def generatePercentChanges(prices):
    ''' Given a price vector, generate a vector of forward price changes. '''
    priceChanges = []

    # Calculate the forward change in price for each pair of days
    for i in range(len(prices)-1):
        priceChanges.append((prices[i+1] - prices[i]) / prices[i])

    return priceChanges



def generateModelData(context, stock, historical_data):
    ''' Given a stock and it's historical data, generate training examples. Each training example is made up of historicalDays
    worth of price changes, and the output variable indicating whether or not the price increased (1), decreased (-1) or no change (0) '''
    inputPriceChanges = []
    outputPrediction = []

    # Generate price changes from historical prices
    priceChanges = generatePercentChanges(historical_data)

    for i in range(len(historical_data) - context.params[stock]["historicalDays"] - context.params[stock]["predictionDays"]):
        # Generate the input training vector from the historicalDays and append to list of training samples
        inputPriceChanges.append( list(priceChanges[i:i + context.params[stock]["historicalDays"] - 1]) )

        # What is this training example's output? Check for a sufficient change in price in either direction
        output = 0
        if historical_data[i + context.params[stock]["historicalDays"] + context.params[stock]["predictionDays"]] > (1+context.params[stock]["percentChange"]) * historical_data[i + context.params[stock]["historicalDays"]]:
            output = 1
        elif historical_data[i + context.params[stock]["historicalDays"] + context.params[stock]["predictionDays"]] < (1-context.params[stock]["percentChange"]) *  historical_data[i + context.params[stock]["historicalDays"]]:
            output = -1
        outputPrediction.append( output )

        # Debugging
        if output != 0: 
            pass
            # log.info("Input tuple: % " % priceChanges[i:i + context.historicalDays - 1])
            # log.info("Output val: %s" % output)
            # log.info("currDay: %f" % historical_data[i + context.historicalDays + context.predictionDays])
            # log.info("gt: %f" % ((1+context.percentChange) * historical_data[i + context.historicalDays]))
    
    # Return the training set
    log.info("training: %s" % outputPrediction)
    return (inputPriceChanges, outputPrediction)