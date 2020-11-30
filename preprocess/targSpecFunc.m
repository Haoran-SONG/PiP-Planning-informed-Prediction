function final_traj = targSpecFunc(traj, tracks)
%% This file only change the target vehicles around the center vehicles
% Now the 25x5 is formed by the relative position of the vehicles
% Then this funciton is used to select those qualified vehicles as
% predication targets, using 7vehs-method.
% So the resulted target number <=8 (7+1following) in each row of traj

gridRows = 25;
gridCols = 5;
nbrStartIdx = 14;
nbrEndIdx   = 13+gridRows * gridCols;
dataNum = size(traj,1);

final_traj = traj;
fprintf('targSpecFunc processing: ')
for i=1:dataNum
    targsGrid = traj(i, nbrStartIdx:nbrEndIdx);
    targsVeh = nonzeros(targsGrid);
    targsNum = length(targsVeh);
    if targsNum<2
        continue;
    end
    dsId = traj(i, 1);
    vehId = traj(i, 2);
    frameId = traj(i, 3);
    centX = traj(i, 4);
    centY = traj(i, 5);
    laneId = traj(i, 6);
    
    %% Retireve the landId and x,y locations of all target vehicles
    targsInfo = zeros(4, targsNum);
    targsInfo(1, :) = targsVeh;
    for j=1:targsNum
        targsInfo(2:4, j) = tracks{dsId, targsVeh(j)}( 2:4, tracks{dsId, targsVeh(j)}(1,:)==frameId );
    end
    
    %% Classify the vehicle of different areas
    % Now the location is transformed to relative position
    targsInfo(2, :) = targsInfo(2, :) - centX;
    targsInfo(3, :) = targsInfo(3, :) - centY;
    precedingInfo = targsInfo(:, (targsInfo(4,:)==laneId) & (targsInfo(3,:)>0) );
    followingInfo = targsInfo(:, (targsInfo(4,:)==laneId) & (targsInfo(3,:)<0) );
    leftLaneInfo  = targsInfo(:, targsInfo(4,:)<laneId );
    rightLaneInfo = targsInfo(:, targsInfo(4,:)>laneId );
    
    %% Pick the qualified vehicles
    qualVehs = zeros(1,8);
    % Vehicle on the same lane
    if size(precedingInfo, 2)==1
        qualVehs(1) = precedingInfo(1);
    elseif size(precedingInfo, 2)>1
        [value, index] = min(precedingInfo(3,:));
        qualVehs(1) = precedingInfo(1, index);
    end
    if size(followingInfo, 2)==1
        qualVehs(2) = followingInfo(1);
    elseif size(followingInfo, 2)>1
        [value, index] = max(followingInfo(3,:));
        qualVehs(2) = followingInfo(1, index);
    end
    % Vehicle on left lane
    leftVehNum = size(leftLaneInfo, 2);
    if leftVehNum==1
        qualVehs(3)   = leftLaneInfo(1);
    elseif leftVehNum==2
        qualVehs(3:4) = leftLaneInfo(1, :);
    elseif leftVehNum>2
        [value, index] = min( leftLaneInfo(2,:).^2 + leftLaneInfo(3,:).^2 );
        qualVehs(3)   = leftLaneInfo(1, index);
        leftCentVehY  = leftLaneInfo(3, index);
        leftFrontVehs = leftLaneInfo(:, leftLaneInfo(3,:)>leftCentVehY);
        if size(leftFrontVehs, 2)>0
            [value, index] = min(leftFrontVehs(3,:));
            qualVehs(4) = leftFrontVehs(1, index);
        end
        leftBackVehs  = leftLaneInfo(:, leftLaneInfo(3,:)<leftCentVehY);
        if size(leftBackVehs, 2)>0
            [value, index] = max(leftBackVehs(3,:));
            qualVehs(5) = leftBackVehs(1, index);
        end
    end
    % Vehicle on right lane
    rightVehNum = size(rightLaneInfo, 2);
    if rightVehNum==1
        qualVehs(6)   = rightLaneInfo(1);
    elseif rightVehNum==2
        qualVehs(6:7) = rightLaneInfo(1, :);
    elseif size(rightLaneInfo, 2)>2
        [value, index] = min( rightLaneInfo(2,:).^2 + rightLaneInfo(3,:).^2 );
        qualVehs(6)    = rightLaneInfo(1, index);
        rightCentVehY  = rightLaneInfo(3, index);
        rightFrontVehs = rightLaneInfo(:, rightLaneInfo(3,:)>rightCentVehY);
        if size(rightFrontVehs, 2)>0
            [value, index] = min(rightFrontVehs(3,:));
            qualVehs(7) = rightFrontVehs(1, index);
        end
        rightBackVehs  = rightLaneInfo(:, rightLaneInfo(3,:)<rightCentVehY);
        if size(rightBackVehs, 2)>0
            [value, index] = max(rightBackVehs(3,:));
            qualVehs(8) = rightBackVehs(1, index);
        end
    end
    
    %% Filter those target vehicles not needed.
    for k = nbrStartIdx:nbrEndIdx
        if traj(i,k)>0 && ~any(qualVehs==traj(i,k))
            final_traj(i,k) = 0;
        end
    end
    
    %Count time------------------------
    if mod(i,100000)==0
        fprintf( '%.2f ... ', i/dataNum);
    end
end

fprintf( '\nPast #dataNum: %d ===>>> Now #dataNum: %d\n', size(traj, 1), size(final_traj, 1));
fprintf( 'Past #targets: %d ===>>> Now #targets: %d\n', nnz(traj(:, nbrStartIdx:nbrEndIdx)), nnz(final_traj(:, nbrStartIdx:nbrEndIdx)));
end