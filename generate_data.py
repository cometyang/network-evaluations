import skimage.transform
import time
import glob
import numpy as np
import mahotas
import random
import matplotlib
import matplotlib.pyplot as plt

def normalizeImage(img, saturation_level=0.05): #was 0.005
	sortedValues = np.sort( img.ravel())
	minVal = np.float32(sortedValues[np.int(len(sortedValues) * (saturation_level / 2))])
	maxVal = np.float32(sortedValues[np.int(len(sortedValues) * (1 - saturation_level / 2))])
	normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))
	normImg[normImg<0] = 0
	normImg[normImg>255] = 255
	return (np.float32(normImg) / 255.0)
	
	
def generate_experiment_data_supervised(purpose='train', nsamples=1000, patchSize=29, balanceRate=0.5, rng=np.random):
    start_time = time.time()

    # quick dirty fix
    #random.seed(rng.rand())

    #pathPrefix = '/media/vkaynig/Data1/Cmor_paper_data/'
    pathPrefix = '/n/pfister_lab/vkaynig/'
    img_search_string_membraneImages = pathPrefix + 'labels/membranes/' + purpose + '/*.tif'
    img_search_string_backgroundMaskImages = pathPrefix + 'labels/background/' + purpose + '/*.tif'
	
    img_search_string_grayImages = pathPrefix + 'images/' + purpose + '/*.tif'
	
    img_files_gray = sorted( glob.glob( img_search_string_grayImages ) )
    img_files_label = sorted( glob.glob( img_search_string_membraneImages ) )
    img_files_backgroundMask = sorted( glob.glob( img_search_string_backgroundMaskImages ) )
	
    whole_set_patches = np.zeros((nsamples, patchSize*patchSize), dtype=np.float)
    whole_set_labels = np.zeros(nsamples, dtype=np.int32)
	
    #how many samples per image?
    nsamples_perImage = np.uint(np.ceil( 
		(nsamples) / np.float(np.shape(img_files_gray)[0])
	)) 
    print 'using ' + np.str(nsamples_perImage) + ' samples per image.'
    counter = 0
	
    img = mahotas.imread(img_files_gray[0])
    grayImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    labelImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    maskImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
	
    for img_index in xrange(np.shape(img_files_gray)[0]):
        img = mahotas.imread(img_files_gray[img_index])
        img = normalizeImage(img) 
        grayImages[:,:,img_index] = img
        label_img = mahotas.imread(img_files_label[img_index])        
        labelImages[:,:,img_index] = label_img
        mask_img = mahotas.imread(img_files_backgroundMask[img_index])
        maskImages[:,:,img_index] = mask_img
		
    for img_index in xrange(np.shape(img_files_gray)[0]):
        img = grayImages[:,:,img_index]        
        label_img = labelImages[:,:,img_index]
        mask_img = maskImages[:,:,img_index]
		
        #get rid of invalid image borders
        border_patch = np.int(np.ceil(patchSize/2.0))
        border = np.int(np.ceil(np.sqrt(2*(border_patch**2))))
        label_img[:border,:] = 0 #top
        label_img[-border:,:] = 0 #bottom
        label_img[:,:border] = 0 #left
        label_img[:,-border:] = 0 #right
		
        mask_img[:border,:] = 0
        mask_img[-border:,:] = 0
        mask_img[:,:border] = 0
        mask_img[:,-border:] = 0
		
        membrane_indices = np.nonzero(label_img)
        non_membrane_indices = np.nonzero(mask_img)
		
        positiveSample = True
        for i in xrange(nsamples_perImage):
            if counter >= nsamples:
                break
            if positiveSample:
                randmem = random.choice(xrange(len(membrane_indices[0])))
                (row,col) = (membrane_indices[0][randmem], 
                             membrane_indices[1][randmem])
                label = 1.0
                positiveSample = False
            else:
                randmem = random.choice(xrange(len(non_membrane_indices[0])))
                (row,col) = (non_membrane_indices[0][randmem], 
                             non_membrane_indices[1][randmem])
                label = 0.0
                positiveSample = True
				
            imgPatch = img[row-border+1:row+border, col-border+1:col+border]
            imgPatch = skimage.transform.rotate(imgPatch, random.choice(xrange(360)))
            imgPatch = imgPatch[border-border_patch:border+border_patch-1,border-border_patch:border+border_patch-1]

            if random.random() < 0.5:
                imgPatch = np.fliplr(imgPatch)
            imgPatch = np.rot90(imgPatch, random.randint(0,3))
                
            whole_set_patches[counter,:] = imgPatch.flatten()
            whole_set_labels[counter] = label
            counter += 1
            
    #normalize data
    whole_data = np.float32(whole_set_patches)
	
    whole_data = whole_data - 0.5
	
    data = whole_data.copy()
    labels = whole_set_labels.copy()
	
    #remove the sorting in image order
    shuffleIndex = rng.permutation(np.shape(labels)[0])
    for i in xrange(np.shape(labels)[0]):  
        whole_data[i,:] = data[shuffleIndex[i],:]
        whole_set_labels[i] = labels[shuffleIndex[i]]
		
    data_set = (whole_data, whole_set_labels)    
    
    end_time = time.time()
    total_time = (end_time - start_time)
    print 'Running time: ' + '%.2fm' % (total_time / 60.)
    rval = data_set
    return rval


def generate_image_data(img, patchSize=29, rows=1):
    img = normalizeImage(img) 

    # pad image borders
    border = np.int(np.ceil(patchSize/2.0))
    img_padded = np.pad(img, border, mode='reflect')

    whole_set_patches = np.zeros((len(rows)*img.shape[1], patchSize**2))

    counter = 0
    for row in rows:
        for col in xrange(img.shape[1]):
            imgPatch = img_padded[row+1:row+2*border, col+1:col+2*border]
            whole_set_patches[counter,:] = imgPatch.flatten()
            counter += 1

    #normalize data
    whole_set_patches = np.float32(whole_set_patches)
    whole_set_patches = whole_set_patches - 0.5

    return whole_set_patches


def stupid_map_wrapper(parameters):
        f = parameters[0]
        args = parameters[1:]
        return f(*args)


# changed the patch sampling to use upper left corner instead of middle pixel
# for patch labels it doesn't matter and it makes sampling even and odd patches easier
def generate_experiment_data_patch_prediction(purpose='train', nsamples=1000, patchSize=29, outPatchSize=1):
    start_time = time.time()

    #pathPrefix = '/media/vkaynig/Data1/Cmor_paper_data/'
    pathPrefix = '/n/pfister_lab/vkaynig/'
    img_search_string_membraneImages = pathPrefix + 'labels/membranes/' + purpose + '/*.tif'
    img_search_string_backgroundMaskImages = pathPrefix + 'labels/background_nonDilate/' + purpose + '/*.tif'

    img_search_string_grayImages = pathPrefix + 'images/' + purpose + '/*.tif'

    img_files_gray = sorted( glob.glob( img_search_string_grayImages ) )
    img_files_label = sorted( glob.glob( img_search_string_membraneImages ) )
    img_files_backgroundMask = sorted( glob.glob( img_search_string_backgroundMaskImages ) )

    whole_set_patches = np.zeros((nsamples, patchSize**2), dtype=np.float)
    whole_set_labels = np.zeros((nsamples, outPatchSize**2), dtype=np.int32)

    #how many samples per image?
    nsamples_perImage = np.uint(np.ceil( 
            (nsamples) / np.float(np.shape(img_files_gray)[0])
            )) 
    print 'using ' + np.str(nsamples_perImage) + ' samples per image.'
    counter = 0

    img = mahotas.imread(img_files_gray[0])
    grayImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    labelImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    maskImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))

    # read the data
    # in random order
    read_order = np.random.permutation(np.shape(img_files_gray)[0])
    for img_index in read_order:
        #print img_files_gray[img_index]
        img = mahotas.imread(img_files_gray[img_index])
        img = normalizeImage(img) 
        grayImages[:,:,img_index] = img
        label_img = mahotas.imread(img_files_label[img_index])        
        labelImages[:,:,img_index] = label_img
        maskImages[:,:,img_index] = 1.0
            
    for img_index in xrange(np.shape(img_files_gray)[0]):
        img = grayImages[:,:,img_index]        
        label_img = labelImages[:,:,img_index]
        mask_img = maskImages[:,:,img_index]

        #get rid of invalid image borders
        mask_img[:,-patchSize:] = 0
        mask_img[-patchSize:,:] = 0

        valid_indices = np.nonzero(mask_img)

        for i in xrange(nsamples_perImage):
            
            if counter >= nsamples:
                break

            randmem = random.choice(xrange(len(valid_indices[0])))
            (row,col) = (valid_indices[0][randmem], 
                         valid_indices[1][randmem])

            imgPatch = img[row:row+patchSize, col:col+patchSize]
            offset_label_patch = int(np.ceil((patchSize - outPatchSize) / 2.0))
            labelPatch = label_img[row+offset_label_patch:row+offset_label_patch+outPatchSize, 
                                   col+offset_label_patch:col+offset_label_patch+outPatchSize]

            if random.random() < 0.5:
                    imgPatch = np.fliplr(imgPatch)
                    labelPatch = np.fliplr(labelPatch)

            rotateInt = random.randint(0,3)
            imgPatch = np.rot90(imgPatch, rotateInt)
            labelPatch = np.rot90(labelPatch, rotateInt)

            whole_set_patches[counter,:] = imgPatch.flatten()
            whole_set_labels[counter] = np.int32(labelPatch.flatten() > 0)
            counter += 1


    #normalize data
    whole_data = np.float32(whole_set_patches)
    whole_data = whole_data - 0.5

    data = whole_data.copy()
    labels = whole_set_labels.copy()

    #remove the sorting in image order
    shuffleIndex = np.random.permutation(np.shape(labels)[0])
    for i in xrange(np.shape(labels)[0]):  
        whole_data[i,:] = data[shuffleIndex[i],:]
        whole_set_labels[i,:] = labels[shuffleIndex[i],:]
    
    data_set = (whole_data, whole_set_labels)    

    end_time = time.time()
    total_time = (end_time - start_time)
    print 'Running time: ', total_time / 60.

    rval = data_set
    print 'finished sampling data'

    return data_set

if __name__=="__main__":
    #data_val = generate_experiment_data_supervised(purpose='validate', nsamples=10000, patchSize=65, balanceRate=0.5)
    data = generate_experiment_data_patch_prediction(purpose='test', nsamples=100, patchSize=29, outPatchSize=29)

