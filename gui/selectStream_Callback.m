function index = selectStream_Callback(streamObj)
index = streamObj.container.findItem(streamObj.uuid);
SyncEEGWithMoBILAB(index);
