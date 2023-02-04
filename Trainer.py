

# 2023-2-3 written by H.Zhang.


import numpy

# ------------------------------------------------------------------------------------- train model.

def train_step(model, features, loss_function, optimizer, labels):

    predictions = model.forward(features)
    loss = loss_function(predictions, labels)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return loss.item()



def train_model(model, loss_function, optimizer, dl_train, epochs = 50):
    
    for epoch  in range(1, epochs+1):
        list_loss = []
        for features, labels in dl_train:
            lossi = train_step(model, features, loss_function, optimizer, labels)
            list_loss.append(lossi)
        loss = numpy.mean(list_loss)
        if epoch % 1 == 0:
            print('epoch={} | loss={} '.format(epoch,loss))




