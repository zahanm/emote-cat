load input.txt;
pidx = randperm(size(input,1));
r_input = input(pidx,:);
labels = r_input(:,1);
features = r_input(:,2:end);
