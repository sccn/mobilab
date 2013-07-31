classdef hardwareMetaData
    properties 
        name
        sampling_rate
        data_items
        count
        label
        bytes = 4;
        uV = [];
        item_size
        bytesHeader
        offset
        reserved_bytes
        originalTimeStamp
    end
    methods
        function obj = hardwareMetaData(varargin)
            if nargin < 1
                obj.name = '';
            elseif ~ischar(varargin{1})
                error('First argument must be a string.');
            else
                obj.name = varargin{1};
            end
            
            if nargin < 2
                obj.sampling_rate = [];
            elseif ~isnumeric(varargin{2})
                error('First argument must be an integer.');
            else
                obj.sampling_rate = varargin{2};
            end
            
            if nargin < 3
                obj.data_items = [];
            elseif ~isnumeric(varargin{3})
                error('First argument must be an integer.');
            else
                obj.data_items = varargin{3};
            end
            
            if nargin < 4
                obj.count = [];
            elseif ~isnumeric(varargin{4})
                error('First argument must be an integer.');
            else
                obj.count = varargin{4};
            end
            
            if nargin < 5
                obj.uV = 1;
            elseif ~isnumeric(varargin{5})
                error('First argument must be an integer.');
            else
                obj.uV = varargin{5};
            end
            
            if nargin < 6
                obj.item_size = [];
            elseif ~isnumeric(varargin{6})
                error('First argument must be an integer.');
            else
                obj.item_size = varargin{6};
            end
            
            if nargin < 7
                obj.bytesHeader = [];
            elseif ~isnumeric(varargin{7})
                error('First argument must be an integer.');
            else
                obj.bytesHeader = varargin{7};
            end
            
            if nargin < 8
                obj.reserved_bytes = [];
            elseif ~isnumeric(varargin{8})
                error('First argument must be an integer.');
            else
                obj.reserved_bytes = varargin{8};
            end
            
            if nargin < 9
                obj.originalTimeStamp = [];
            elseif ~isnumeric(varargin{9})
                error('First argument must be an integer.');
            else
                obj.originalTimeStamp = varargin{9};
            end
            
            if length(obj.uV) > length(obj.count) && obj.uV(1) == 1
                obj.uV(1) = [];
            end 
            if isempty(obj.uV), obj.uV = 1;end
        end
        %%
        function xmlStream = insertInHeader(obj,docNode)
            if nargin < 2, error('Not enough input arguments.');end
            
            bytesInstr = num2str(obj.bytes);
            
            xmlStream = docNode.createElement('stream');
            xmlStream.setAttribute('name',obj.name);
            if ~isempty(obj.bytesHeader), xmlStream.setAttribute('bytes',num2str(obj.bytesHeader));end
            xmlStream.setAttribute('sampling_rate',num2str(obj.sampling_rate));
            
            timestamp = docNode.createElement('timestamp');
            timestamp.setAttribute('bytes',bytesInstr);
            xmlStream.appendChild(timestamp);
            
            event = docNode.createElement('event');
            event.setAttribute('bytes',bytesInstr);
            xmlStream.appendChild(event);

            data_items = docNode.createElement('data_items'); %#ok
            data_items.setAttribute('bytes',bytesInstr);      %#ok
            xmlStream.appendChild(data_items);                %#ok  
            
            item_size = docNode.createElement('item_size');   %#ok
            item_size.setAttribute('bytes',bytesInstr);       %#ok
            item_size.setAttribute('value',num2str(obj.item_size));     %#ok
            xmlStream.appendChild(item_size);                    %#ok
            
            data = docNode.createElement('data');
            data.setAttribute('bytes',num2str(obj.data_items*obj.item_size));
            data.setAttribute('items',num2str(obj.data_items));
            switch obj.name
                case 'biosemi'
                    for it=1:length(obj.label)
                        dataItem = docNode.createElement(obj.label{it}(1:end-1));
                        dataItem.setAttribute('uV',num2str(obj.uV(it)));
                        dataItem.setAttribute('count',num2str(obj.count(it)));
                        labelItem = docNode.createElement('ch');
                        labelItem.setAttribute('label',obj.label{it});
                        dataItem.appendChild(labelItem);
                        data.appendChild(dataItem);
                    end
                case 'phasespace'
                    marker = docNode.createElement('marker');
                    marker.setAttribute('count',num2str(obj.count));
                    for it=1:length(obj.label)
                        labelItem = docNode.createElement(obj.label{it}(1:end-1));
                        labelItem.setAttribute('label',obj.label{it});
                        marker.appendChild(labelItem);
                    end
                    data.appendChild(marker);
                case 'wii'
                    wii_acc = docNode.createElement('wii_acc');
                    wii_acc.setAttribute('count',num2str(obj.count(1)));
                    for it=1:3
                        labelItem = docNode.createElement(obj.label{it}(1:end-1));
                        labelItem.setAttribute('label',obj.label{it});
                        wii_acc.appendChild(labelItem);
                    end
                    data.appendChild(wii_acc);
                    wii_IR = docNode.createElement('wii_IR');
                    wii_IR.setAttribute('count',num2str(obj.count(1)));
                    for it=4:9
                        labelItem = docNode.createElement(obj.label{it}(1:end-1));
                        labelItem.setAttribute('label',obj.label{it});
                        wii_IR.appendChild(labelItem);
                    end
                    data.appendChild(wii_IR);
                case 'eventcode'
                
                otherwise
            end
            xmlStream.appendChild(data);
        end
    end
end