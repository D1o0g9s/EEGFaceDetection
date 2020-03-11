from helperFunctions import *
from constants import *


#after reading in file, pass in data.
#returns 2 length array, with index 0 indicating if image is a face; index 2 is eeg data regarding attention, etc.
def filterDataForPipline(data):
    trialData = getMarkerBoundSingleMarkerData('NewImageStart', 'NewImageEnd', data, go_backward=False)
    allFilteredData = []
    #print(len(trialData[0]))
    for i in range(len(trialData[0])):
        filterData = [0, []]
        if 'Face' in trialData[0][i].get('markers').get('data'):
            filterData[0] = 1
        filterData[1] = trialData[0][i].get('eeg').get('data')
        allFilteredData.append(filterData)
        
    return allFilteredData


##Example on how to use:    

# filename2 = "part_P002_block_S001"

# XDF_Path2 = "./data/"+filename2+".xdf"
# XDF_Data2 = loadxdf(XDF_Path2)

# XDF_Path3 = "part_P003_block_S001"
# XDF_Path3 = "./data/"+filename3+".xdf"
# XDF_Data3 = loadxdf(XDF_Path3)

# filename1 = "part_P001_block_S001"
# XDF_Path1 = "./data/"+filename1+".xdf"
# XDF_Data1 = loadxdf(XDF_Path1)


# filteredData1Pipline = filterDataForPipline(XDF_Data1)

# filteredData2Pipline = filterDataForPipline(XDF_Data2)

# filteredData3Pipline = filterDataForPipline(XDF_Data3)