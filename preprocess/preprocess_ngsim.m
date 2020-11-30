%% Process NGSIM dataset into required datasets %%
clear;
clc;


%% WORKFLOW
%{
1) Reading csv files 
2) Parse fields including sptail grid and maneuver labels
3) Using unique vehicle ids to spilit train(70%)/validation(10%)/test(20%)
4) Only reserve those data sample with at least 3s history and 5s future
5) Save the dataset with a fixed 8Veh targets
Optional: filter on-ramp and off-ramp part or not. (Our result is obtained without filtering lane)
%}

%% Hyperparameters:
%{
30ms for history traj
50ms for future traj
past and future 40ms for lateral behavior detaction.
grid is splitted with size 25*5 (8x7 feet for each grid)
                           25*7 (8x5 feet for each grid)
%}

lane_filter = false;
grid_length=25; grid_width=5; cell_length=8; cell_width=7;

% Save location
if lane_filter
    raw_folder = sprintf('./dataset/ngsim/%dx%d_raw/', grid_length, grid_width);
    post_folder = sprintf('./dataset/ngsim/%dx%d/', grid_length, grid_width);
    fix_tar_folder = sprintf('./dataset/ngsim/%dx%d_8Veh/', grid_length, grid_width);
else
    raw_folder = sprintf('./dataset/ngsim/%dx%d_nofL_raw/', grid_length, grid_width);
    post_folder = sprintf('./dataset/ngsim/%dx%d_nofL/', grid_length, grid_width);
    fix_tar_folder = sprintf('./dataset/ngsim/%dx%d_8Veh_nofL/', grid_length, grid_width);
end    
mkdir(raw_folder); 
mkdir(post_folder);
mkdir(fix_tar_folder);

% Other variable dependent on grid.
grid_cells = grid_length * grid_width;
grid_cent_location = ceil(grid_length*grid_width*0.5);

%% Fields in the final result:
% traj  : (data number)*(13+grid_num)
%{
1: Dataset Id
2: Vehicle Id
(Column in tracks)
|3 : Frame Id
|4 : Local X
|5 : Local Y
|6 : Lane Id
|7*: Lateral maneuver
|8*: Longitudinal maneuver
|9 : Length
|10: Width
|11: Class label
|12: Velocity
|13: Accerlation
|14-end*: Neighbor Car Ids at grid location
%}

% tracks: includes {Dataset_Id*Vehicle_Id}, each cell with (11+grid)*totalFramNum
%{
|1 : Frame Id
|2 : Local X
|3 : Local Y
|4 : Lane Id
|5 : Lateral maneuver
|6 : Longitudinal maneuver
|7*: Length
|8*: Width
|9 : Class label
|10: Velocity
|11: Accerlation
|12-end*: Neighbor Car Ids at grid location
%}


%% 0.Inputs: Locations of raw_ngsim input files:
dataset_to_use = 6;
us101_1 = './raw_ngsim/us101-0750am-0805am.txt';
us101_2 = './raw_ngsim/us101-0805am-0820am.txt';
us101_3 = './raw_ngsim/us101-0820am-0835am.txt';
i80_1 = './raw_ngsim/i80-0400-0415.txt';
i80_2 = './raw_ngsim/i80-0500-0515.txt';
i80_3 = './raw_ngsim/i80-0515-0530.txt';


%% 1.Load data 
disp('Loading data...')
% Add dataset id at the 1st column
traj{1} = load(us101_1);    
traj{1} = single([ones(size(traj{1},1),1),traj{1}]);
traj{2} = load(us101_2);
traj{2} = single([2*ones(size(traj{2},1),1),traj{2}]);
traj{3} = load(us101_3);
traj{3} = single([3*ones(size(traj{3},1),1),traj{3}]);
traj{4} = load(i80_1);    
traj{4} = single([4*ones(size(traj{4},1),1),traj{4}]);
traj{5} = load(i80_2);
traj{5} = single([5*ones(size(traj{5},1),1),traj{5}]);
traj{6} = load(i80_3);
traj{6} = single([6*ones(size(traj{6},1),1),traj{6}]);

% traj is the base for adding more infomation later. 
for k = 1:dataset_to_use
    % Loading 1:dataset id, 2:Vehicle id, 3:Frame index, 
    %         6:Local X, 7:Local Y, 15:Lane id.
    %         10:v_length, 11:v_Width, 12:v_Class
    %         13:Velocity (feet/s), 14:Acceleration (feet/s2).
    traj{k} = traj{k}(:,[1,2,3,6,7,15,10,11,12,13,14]);
    
    % @@ Filter all vehicles in the parts of on-ramp and off-ramp
    if lane_filter
        fprintf( 'Dataset-%d #data: %d ==>> ', k, size(traj{k}, 1));
        traj{k} = traj{k}(traj{k}(:, 6) < 7, :);
        fprintf( '%d after filtering lane>6 \n', size(traj{k}, 1));
    else    
        % Prev: US101 make all lane id >= 6 to 6.
        if k <=3
            traj{k}( traj{k}(:,6)>=6,6 ) = 6;  
        end
    end
    % Leave space for maneuver labels (2 columns) and grid (grid_cells columns)
    traj{k} = [ traj{k}(:,1:6), zeros(size(traj{k},1),2), traj{k}(:,7:11), zeros(size(traj{k},1),grid_cells) ];
end

% Use the vehilce's center as its location
offset = zeros(1,dataset_to_use);
for k = 1:dataset_to_use
    traj{k}(:,5) = traj{k}(:,5) - 0.5*traj{k}(:,9);
    offset(k) = min(traj{k}(:,5));
    if offset(k) < 0
        % To make the Y location > 0
        traj{k}(:,5) = traj{k}(:,5) - offset(k);
    end
end


%% 2.Parse fields (listed above: maneuver label, neighbour grid)
disp('Parsing fields...')

poolobj = parpool(dataset_to_use);
parfor ii = 1:dataset_to_use   % Loop on each dataset.
% for ii = 1:dataset_to_use
    tic;
    disp(['Now process dataset ', num2str(ii)])
   
    % Loop on each row.
    for k = 1:length(traj{ii}(:,1)) 
        % Refresh the process every 1 mins
        if toc > 60  
            fprintf( 'Dataset-%d: Complete %.3f%% \n', ii, k/length(traj{ii}(:,1))*100 );
            tic;
        end

        dsId = ii;
        vehId = traj{ii}(k,2);
        time = traj{ii}(k,3);
        % Get all rows about this vehId
        vehtraj = traj{ii}(traj{ii}(:,2)==vehId, : );  
    
        % Get the row index of traj at this frame.
        ind = find(vehtraj(:,3)==time);                                      
        ind = ind(1);
        lane = traj{ii}(k,6);
        
        % Lateral maneuver in Column 7:
        ub = min(size(vehtraj,1),ind+40);                                %Upper boundary (+40 frame)
        lb = max(1, ind-40);                                             %Lower boundary (-40 frame)
        if vehtraj(ub,6)>vehtraj(ind,6) || vehtraj(ind,6)>vehtraj(lb,6)  %(prepate to turn or stablize after turn)
            traj{ii}(k,7) = 3;   % Turn Right==>3. 
        elseif vehtraj(ub,6)<vehtraj(ind,6) || vehtraj(ind,6)<vehtraj(lb,6)
            traj{ii}(k,7) = 2;   % Turn Left==>2.
        else
            traj{ii}(k,7) = 1;   % Keep lane==>1.
        end
        
        % Longitudinal maneuver in Column 8:
        ub = min(size(vehtraj,1),ind+50); % Future boundary  (+50 frame)
        lb = max(1, ind-30);              % History boundary (-30 frame)
        if ub==ind || lb ==ind
            traj{ii}(k,8) = 1;   % Normal==>1
        else
            vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb);
            vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);
            if vFut/vHist <0.8
                traj{ii}(k,8) = 2; % Brake==> 2
            else
                traj{ii}(k,8) = 1; % Normal==>1
            end
        end
        
        % Get grid locations in Column 14~13+grid_length*grid_width (grid_cells, each with cell_length*cell_width): 
        centVehX = traj{ii}(k,4);
        centVehY = traj{ii}(k,5);
        gridMinX = centVehX - 0.5*grid_width*cell_width;
        gridMinY = centVehY - 0.5*grid_length*cell_length;
        otherVehsAtTime = traj{ii}( traj{ii}(:,3)==time , [2,4,5]);  % Only keep the (vehId, localX, localY)        
        otherVehsInSizeRnage = otherVehsAtTime( abs(otherVehsAtTime(:,3)-centVehY)<(0.5*grid_length*cell_length) ...
                                              & abs(otherVehsAtTime(:,2)-centVehX)<(0.5*grid_width*cell_width) , :);
        if ~isempty(otherVehsInSizeRnage)
            % Lateral and Longitute grid location. Finally exact location is saved in the 3rd column;
            otherVehsInSizeRnage(:,2) = ceil((otherVehsInSizeRnage(:,2) - gridMinX) / cell_width); 
            otherVehsInSizeRnage(:,3) = ceil((otherVehsInSizeRnage(:,3) - gridMinY) / cell_length); 
            otherVehsInSizeRnage(:,3) = otherVehsInSizeRnage(:,3) + (otherVehsInSizeRnage(:,2)-1) * grid_length; 
            for l = 1:size(otherVehsInSizeRnage, 1)
                exactGridLocation = otherVehsInSizeRnage(l,3);
                if exactGridLocation ~= grid_cent_location % The center gird location is kept to NONE
                    traj{ii}(k,13+exactGridLocation) = otherVehsInSizeRnage(l,1);
                end
            end   
        end
        
    end
end
delete(poolobj);


%% 3.Merge and Split train, validation, test
disp('Splitting into train, validation and test sets...')

% Merge all datasets together.
trajAll = [];
for i = 1:dataset_to_use
    trajAll = [trajAll; traj{i}];
    fprintf( 'Now merge %d rows of data from traj{%d} \n', size(traj{i},1), i);
end
clear traj;

% Training, Validation and Test dataset (Everything together)
trajTr = [];
trajVal = [];
trajTs = [];
for k = 1:dataset_to_use  % Split the vehilce's trajectory in all dataset_to_use
    % Cutting point: 0.7*max vehilecId (70% Training set)
    ul1 = round(0.7* max( trajAll(trajAll(:,1)==k,2) ));  
    % Cutting point: 0.8*max vehilecId (20% Test set)
    ul2 = round(0.8* max( trajAll(trajAll(:,1)==k,2) ));  
    % Extract according to the vehicle ID 
    trajTr =  [trajTr;  trajAll(trajAll(:,1)==k & trajAll(:,2)<=ul1, :) ]; 
    trajVal = [trajVal; trajAll(trajAll(:,1)==k & trajAll(:,2)>ul1 & trajAll(:,2)<=ul2, :) ];
    trajTs =  [trajTs;  trajAll(trajAll(:,1)==k & trajAll(:,2)>ul2, :) ];
end

% Merging all info together in tracks
% The neighbour existence problem is addressed
tracks = {};
for k = 1:dataset_to_use
    trajSet = trajAll(trajAll(:,1)==k,:); 
    carIds = unique(trajSet(:,2));      % Unique Vehicle ID, then get a cell for each car.    
    for l = 1:length(carIds)
        % The cell in {datasetID, carID} is placed with (11+grid_cells)*TotalFram.
        tracks{k,carIds(l)} = trajSet( trajSet(:,2)==carIds(l), 3:end )';                      
    end
end

%% 4.Filter edge cases: 
disp('Filtering edge cases...')

% Flag for whether to discard this row of dataraw_folder
indsTr = zeros(size(trajTr,1),1); 
indsVal = zeros(size(trajVal,1),1);
indsTs = zeros(size(trajTs,1),1);

% Since the model uses 3 sec of trajectory history for prediction, and 5s
% future for planning, therefore the reserve condition for each row of data: 
    % 1) this frame t should be larger than the 31st id, and
    % 2) has at least 5s future.
    
for k = 1: size(trajTr,1)   % Loop on each row of traj.
    t = trajTr(k,3);                        
    if tracks{trajTr(k,1),trajTr(k,2)}(1,31) <= t && tracks{trajTr(k,1),trajTr(k,2)}(1,end)>= t+50
        indsTr(k) = 1;
    end
end
trajTr = trajTr(find(indsTr),:);
for k = 1: size(trajVal,1)
    t = trajVal(k,3);
    if tracks{trajVal(k,1),trajVal(k,2)}(1,31) <= t && tracks{trajVal(k,1),trajVal(k,2)}(1,end)>= t+50
        indsVal(k) = 1;
    end
end
trajVal = trajVal(find(indsVal),:);
for k = 1: size(trajTs,1)
    t = trajTs(k,3);
    if tracks{trajTs(k,1),trajTs(k,2)}(1,31) <= t && tracks{trajTs(k,1),trajTs(k,2)}(1,end)>= t+50
        indsTs(k) = 1;
    end
end
trajTs = trajTs(find(indsTs),:);


%% Save mat files:
% traj  : n*(13+grid_cells), n is the data number.
% tracks: 6*maxVehicleId, each cell is specified for (datasetId, vehicleId), with size (11+grid_cells)*totalFramNum.
disp('Saving mat files...')

% Save raw data
save(strcat(raw_folder,'gridTrainAround'), 'trajTr', 'tracks','-v7.3');
save(strcat(raw_folder,'gridValAround'),  'trajVal','tracks','-v7.3');
save(strcat(raw_folder,'gridTestAround'), 'trajTs', 'tracks','-v7.3');

% Save post-processed data
fprintf( '### Train data: \n');
traj = nbhCheckerFunc(trajTr, tracks);
save(strcat(post_folder,'gridTrainAround'),'traj','tracks','-v7.3');
traj = targSpecFunc(traj, tracks);
save(strcat(fix_tar_folder,'gridTrainAround'),'traj','tracks','-v7.3');

fprintf( '### Validation data: \n');
traj = nbhCheckerFunc(trajVal, tracks);
save(strcat(post_folder,'gridValAround'),'traj','tracks','-v7.3');
traj = targSpecFunc(traj, tracks);
save(strcat(fix_tar_folder,'gridValAround'),'traj','tracks','-v7.3');

fprintf( '### Test data: \n');
traj = nbhCheckerFunc(trajTs, tracks);
save(strcat(post_folder,'gridTestAround'),'traj','tracks','-v7.3');
traj = targSpecFunc(traj, tracks);
save(strcat(fix_tar_folder,'gridTestAround'),'traj','tracks','-v7.3');

fprintf('Complete');