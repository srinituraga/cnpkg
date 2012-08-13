function m = fix_malis_step_no(m)

s = m.step_map.gradient
if length(s) > 1, error('step length is greater than 1!!'), end
s = [s s+1];

for k = 1:length(m.layers),
	if m.layers{k}.stepNo == m.step_map.gradient,
		m.layers{k}.stepNo = s;
	end
end
m.step_map.gradient = s;
