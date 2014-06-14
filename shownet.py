import matplotlib
matplotlib.use('Agg') # Don't display
import matplotlib.pyplot as plt
import gfx
import numpy as np

def plotCost(fname, train=None, valid=None, test=None,
             train_label='Train', test_label='Test'):
    ''' Plot a dual axis plot that shows training costs and validation
        accuracy superimposed. '''
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')

    if train is not None:
        if len(train) < 60:
            ax1.plot(train, 'bo-')
        else:
            ax1.plot(train, 'b-')
        ax1.set_ylabel(train_label)

    if valid is not None:
        valid_x = np.linspace(0,len(train)-1,len(valid))
        if len(valid) < 60:
            ax1.plot(valid_x, valid, 'go-')
        else:
            ax1.plot(valid_x, valid, 'g-')

    if test is not None:
        ax2 = ax1.twinx()
        test_x = np.linspace(0,len(train)-1,len(test))
        if len(test) < 60:
            ax2.plot(test_x, test, 'ro-')
        else:
            ax2.plot(test_x, test, 'r-')
        ax2.set_ylabel(test_label, color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

    plt.savefig(fname, bbox_inches='tight')
    gfx.render(fname)
    plt.close()

def make_filter_fig(fname, filters, filter_start=0, num_filters=16*16, 
                    _title='', combine_chans=False):
    ''' 
    Adapated from cuda-convnet
    fname: filename to save to
    filters: the weights - a 2-3 Tensor: [colors][from weights][to weights]
    filter_start: number of the filter to visualize first
    num_filters: number of filters to visualize, capped at 16*16
    _title: figure title
    combine_chans: combine rgb channels
    '''

    if len(filters.shape) == 2:
        filters = np.expand_dims(filters, 0)

    FILTERS_PER_ROW = 16
    MAX_ROWS = 16
    MAX_FILTERS = min(FILTERS_PER_ROW * MAX_ROWS, filters.shape[2])
    num_colors = filters.shape[0]
    f_per_row = int(np.ceil(FILTERS_PER_ROW / float(1 if combine_chans else num_colors)))
    filter_end = min(filter_start+MAX_FILTERS, num_filters)
    filter_rows = int(np.ceil(float(filter_end - filter_start) / f_per_row))

    filter_size = int(np.sqrt(filters.shape[1]))
    fig = plt.figure()
    fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
    num_filters = filter_end - filter_start
    if not combine_chans:
        bigpic = np.zeros((filter_size * filter_rows + filter_rows + 1,
                           filter_size*num_colors * f_per_row + f_per_row + 1),
                          dtype=np.single)
    else:
        bigpic = np.zeros((3, filter_size * filter_rows + filter_rows + 1,
                           filter_size * f_per_row + f_per_row + 1),
                          dtype=np.single)

    for m in xrange(filter_start,filter_end):
        filter = filters[:,:,m]
        y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
        if not combine_chans:
            for c in xrange(num_colors):
                filter_pic = filter[c,:].reshape((filter_size,filter_size))
                bigpic[1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                       1 + (1 + filter_size*num_colors) * x + filter_size*c:1 + (1 + filter_size*num_colors) * x + filter_size*(c+1)] = filter_pic
        else:
            filter_pic = filter.reshape((3, filter_size,filter_size))
            bigpic[:,
                   1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                   1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic

    plt.xticks([])
    plt.yticks([])
    if not combine_chans:
        plt.imsave(fname, bigpic, cmap=plt.cm.gray)#, interpolation='nearest')
    else:
        bigpic = bigpic.swapaxes(0,2).swapaxes(0,1)
        plt.imsave(fname, bigpic) #, interpolation='nearest')
    gfx.render(fname)
    plt.close()


def plot_confusion_matrix(fname, y_true, y_pred, labels=None):
    ''' Plots a confusion matrix. '''
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels)
    cm = np.repeat(np.repeat(cm, 20, axis=0), 20, axis=1)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.imsave(fname, cm)
    gfx.render(fname)
    plt.close()
