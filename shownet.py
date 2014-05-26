import matplotlib
matplotlib.use('Agg') # Don't display
import matplotlib.pyplot as plt
import gfx

def plotCost(costs, valid):
    ''' Plot a dual axis plot that shows training costs and validation
        accuracy superimposed. '''
    fig, ax1 = plt.subplots()
    ax1.plot(costs, 'b-')
    ax1.set_xlabel('Epoch')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Cost', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    valid_x = numpy.linspace(0,len(costs),len(valid))
    ax2.plot(valid_x, valid, 'ro-')
    ax2.set_ylabel('Validation Loss', color='r')
    plt.ylim([0,100])
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.savefig('results/train.png')
    gfx.render('results/train.png')
    plt.close()
