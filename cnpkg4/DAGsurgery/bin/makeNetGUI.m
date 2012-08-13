classdef makeNetGUI
	properties
		m;
		dag;
		name;
	end
	methods
		function dn = makeNetGUI(locm)
			%javaaddpath ./DAGedit.jar
			
			dn.name = ['DAG',sprintf('%04d',floor(10000*rand()))];
			dn.dag = DAGFrame(dn.name);
			
			if(exist('locm') && ~isempty(locm))
				dn.m = locm;
				dn.loadm(dn.m);
			end
		end %matDAG
		
		function loadm(dn,mvar)
			%First add activities/inputs, biases
			for i=1:length(mvar.layers)
				if(strcmp(mvar.layers{i}.type,'input'))
					layer_name = mvar.layers{i}.name;
					my_id = int32(mvar.layers{i}.z);
					fm_ct = int32(mvar.layers{i}.size{1});
					if(isfield(mvar.layers{i},'y_space'))
						y_s = int32(mvar.layers{i}.y_space);
					else
						y_s = 1;
					end
					if(isfield(mvar.layers{i},'x_space'))
						x_s = int32(mvar.layers{i}.x_space);
					else
						x_s = 1;
					end
					if(isfield(mvar.layers{i},'y_space'))
						d_s = int32(mvar.layers{i}.d_space);
					else
						d_s=1;
					end
					dn.dag.addUncomputed(layer_name,my_id,fm_ct,y_s,x_s,d_s);
				elseif(strcmp(mvar.layers{i}.type,'hidden') || strcmp(mvar.layers{i}.type,'output') || strcmp(mvar.layers{i}.type,'computed'))
					layer_name = mvar.layers{i}.name;
					fm_ct = int32(mvar.layers{i}.size{1});
					my_id = int32(mvar.layers{i}.z);
					if(isfield(mvar.layers{i},'y_space'))
						y_s = int32(mvar.layers{i}.y_space);
					else
						y_s = 1;
					end
					if(isfield(mvar.layers{i},'x_space'))
						x_s = int32(mvar.layers{i}.x_space);
					else
						x_s = 1;
					end
					if(isfield(mvar.layers{i},'y_space'))
						d_s = int32(mvar.layers{i}.d_space);
					else
						d_s=1;
					end
					bid = mvar.layers{i}.zb;
					linear_bias = single(reshape(mvar.layers{bid}.val,[1,length(mvar.layers{bid}.val)]));
					b_eta = single(mvar.layers{bid}.eta);
					dn.dag.addComputed(layer_name,my_id,fm_ct,linear_bias,y_s,x_s,d_s,b_eta);
				end
			end
			% Then add sensitivities, weights
			for i=1:length(mvar.layers)
				if(strcmp(mvar.layers{i}.type,'sens'))
					z_me = int32(mvar.layers{i}.zme);
					dn.dag.addSens(z_me);
				elseif(strcmp(mvar.layers{i}.type,'error'))
					z_me = int32(mvar.layers{i}.zme);
					dn.dag.addErr(z_me);
				elseif(strcmp(mvar.layers{i}.type,'hidden') || strcmp(mvar.layers{i}.type,'output') || strcmp(mvar.layers{i}.type,'computed'))
					to = int32(mvar.layers{i}.z);
					wids = mvar.layers{i}.zw;
					for j=1:length(wids)
						layer_name = mvar.layers{wids(j)}.name;
						my_id = int32(mvar.layers{wids(j)}.z);
						s_mi = int32(mvar.layers{wids(j)}.size{1});
						s_x = int32(mvar.layers{wids(j)}.size{2});
						s_y = int32(mvar.layers{wids(j)}.size{3});
						s_z = int32(mvar.layers{wids(j)}.size{4});
						s_mo = int32(mvar.layers{wids(j)}.size{5});
						linear_vals = single(reshape(mvar.layers{wids(j)}.val,[1,numel(mvar.layers{wids(j)}.val)]));
						if(isfield(mvar.layers{wids(j)},'y_space'))
							y_s = int32(mvar.layers{wids(j)}.y_space);
						else
							y_s = 1;
						end
						if(isfield(mvar.layers{wids(j)},'x_space'))
							x_s = int32(mvar.layers{wids(j)}.x_space);
						else
							x_s = 1;
						end
						if(isfield(mvar.layers{wids(j)},'y_space'))
							d_s = int32(mvar.layers{wids(j)}.d_space);
						else
							d_s=1;
						end
						if(isfield(mvar.layers{wids(j)},'wPxSize'))
							y_b = int32(mvar.layers{wids(j)}.wPxSize(1));
							x_b = int32(mvar.layers{wids(j)}.wPxSize(2));
							d_b = int32(mvar.layers{wids(j)}.wPxSize(3));
						else
							if(isfield(mvar.layers{wids(j)},'y_blk'))
								y_b = int32(mvar.layers{wids(j)}.y_blk);
							else
								y_b=1;
							end
							if(isfield(mvar.layers{wids(j)},'x_blk'))
								x_b = int32(mvar.layers{wids(j)}.x_blk);
							else
								x_b=1;
							end
							if(isfield(mvar.layers{wids(j)},'d_blk'))
								d_b = int32(mvar.layers{wids(j)}.d_blk);
							else
								d_b=1;
							end
						end
						w_eta = single(mvar.layers{wids(j)}.eta);
						from = int32(mvar.layers{wids(j)}.zp);
						dn.dag.addWeight(from,to,layer_name,my_id,s_x,s_y,s_z,linear_vals,x_s,y_s,d_s,x_b,y_b,d_b,w_eta);
					end
				end
			end
			dn.dag.makeClean();
		end
		
		function clear(dn)
			dn.dag.dispose();
		end
		
		function m2 = savem(dn)
			dn.dag.cleanup();
			mapped_to = [];
			layers={};
			num_uncomputed = dn.dag.numUncomputed();
			
			inputs=[];
			computed={};
			sens={};
			outputs=[];
			errors=[];
			labels=[];
			compprev={};
			compnext={};
			sensnext={};
			weight={};
			bias={};
			weightout={};
			
			z=1;
			layers{z}=struct('name',{'xi'},'type',{'index'},'size',{{[5],[0],[1]}},'val',{0},'stepNo',{[]});
			z = z+1;
			sens_forp=[];
			sens_for=[];
			for i=1:num_uncomputed
				layer_name = dn.dag.getUncomputedName(int32(i-1));
				fm_ct = dn.dag.getUncomputedMapCt(int32(i-1));
				space = dn.dag.getUncomputedSpace(int32(i-1));
				id = dn.dag.getUncomputedId(int32(i-1));
				mapped_to(id+1)=z;
				sens_for(id+1)=-1;
				sens_forp(z)=-1;
				layers{z}=struct('z',{z},'name',{layer_name'},'type',{'input'},'size',{{[double(fm_ct)],[double(0)],[double(0)],[double(0)],[double(1)]}},'y_space',{double(space(2))},'x_space',{double(space(1))},'d_space',{double(space(3))},'y_start',{[double(0)]},'x_start',{[double(0)]},'d_start',{[double(0)]},'zin',{1},'zn',{[]},'stepNo',{[]},'val',{0});	
				inputs(end+1) = z;
				z = z+1;
			end
			num_computed = dn.dag.numComputed();
			
			for i=1:num_computed
				layer_name = dn.dag.getComputedName(int32(i-1));
				fm_ct = dn.dag.getComputedMapCt(int32(i-1));
				space = dn.dag.getComputedSpace(int32(i-1));
				id = dn.dag.getComputedId(int32(i-1));
				layers{z}=struct('z',{z},'name',{layer_name'},'type',{'computed'},'zp',{[]},'zw',{[]},'zb',{[z+1]},'zn',{[]},'size',{{[double(fm_ct)],[double(0)],[double(0)],[double(0)],[double(1)]}},'y_space',{double(space(2))},'x_space',{double(space(1))},'d_space',{double(space(3))},'y_start',{[double(0)]},'x_start',{[double(0)]},'d_start',{[double(0)]},'stepNo',{[]},'val',{0});
				computed{end+1}=z;
				mapped_to(id+1)=z;
				sens_for(id+1)=-1;
				sens_forp(z)=-1;
				z=z+1;
				b_vals = dn.dag.getComputedBias(int32(i-1));
				b_eta = dn.dag.getComputedEta(int32(i-1));
				layers{z}=struct('z',{z},'name',{['bias_',layer_name']},'type',{'bias'},'eta',{b_eta},'zs',{[]},'size',{{[double(fm_ct)],[double(1)]}},'val',{b_vals},'dval',{zeros(size(b_vals))},'stepNo',{[]});
				bias{end+1}={z};
				z=z+1;
				if dn.dag.computedHasSens(int32(i-1))
					layers{z-1}.zs=z;
					layers{z}=struct('z',{z},'name',{['sens_',layer_name']},'type',{'sens'},'zme',{z-2},'znw',{[]},'zn',{[]},'size',{{[double(fm_ct)],[double(0)],[double(0)],[double(0)],[double(1)]}},'y_space',{double(space(2))},'x_space',{double(space(1))},'d_space',{double(space(3))},'y_start',{[double(0)]},'x_start',{[double(0)]},'d_start',{[double(0)]},'stepNo',{[]},'val',{0});
					sens{end+1}=z;
					sens_for(id+1)=z;
					sens_forp(z-2)=z;
					layers{z-1}.zs = z;
					z=z+1;
				elseif dn.dag.computedHasErr(int32(i-1))
				    layers{z-1}.zs = z;
					layers{z}=struct('z',{z},'name',{['error_',layer_name']},'type',{'error'},'zme',{z-2},'zn',{[]},'znw',{[]},'size',{{[double(fm_ct)],[double(0)],[double(0)],[double(0)],[double(1)]}},'zl',{z+1},'y_space',{double(space(2))},'x_space',{double(space(1))},'d_space',{double(space(3))},'y_start',{[double(0)]},'x_start',{[double(0)]},'d_start',{[double(0)]},'stepNo',{[]},'val',{0},'loss',{0},'classerr',{0});
					sens{end+1}=z;
					sens_for(id+1)=z;
					sens_forp(z-2)=z;
					layers{z-1}.zs = z;
					z=z+1;
					layers{z}=struct('z',{z},'name',{['label_',layer_name']},'type',{'label'},'zin',{1},'size',{{[double(fm_ct)],[double(0)],[double(0)],[double(0)],[double(1)]}},'y_space',{double(space(2))},'x_space',{double(space(1))},'d_space',{double(space(3))},'y_start',{[double(0)]},'x_start',{[double(0)]},'d_start',{[double(0)]},'stepNo',{[]},'val',{0},'mask',{0});
					z=z+1;
				else
					sens{end+1}=[];
				end
			end
			
			num_conn = dn.dag.numConn();
			
			for i=1:num_conn
				layer_name = dn.dag.getConnName(int32(i-1));
				from = dn.dag.getConnFrom(int32(i-1));
				to = dn.dag.getConnTo(int32(i-1));
				wsize = dn.dag.getConnSize(int32(i-1));
				wvals = dn.dag.getConnVals(int32(i-1));
				space = dn.dag.getConnSpace(int32(i-1));
				block = dn.dag.getConnBlock(int32(i-1));
				weta = dn.dag.getConnEta(int32(i-1));
				wvals = reshape(wvals,[wsize(1),wsize(2),wsize(3),wsize(4),wsize(5)]);
				layers{z}=struct('z',{z},'name',{layer_name'},'type',{'weight'},'convType',{'valid'},'zp',{mapped_to(from+1)},'zs',{[]},'eta',{weta},'y_blk',{double(block(2))},'x_blk',{double(block(1))},'d_blk',{double(block(3))},'y_space',{double(space(2))},'x_space',{double(space(1))},'d_space',{double(space(3))},'size',{{[double(wsize(1))],[double(wsize(2))],[double(wsize(3))],[double(wsize(4))],[double(wsize(5))]}},'val',{wvals},'dval',{zeros(size(wvals))},'stepNo',{[]});
				layers{mapped_to(to+1)}.zw(end+1) = z;
				layers{mapped_to(to+1)}.zp(end+1) = mapped_to(from+1);
				%sens for mapped_to(from) has znw as this and zn as this's sens
				
				layers{mapped_to(from+1)}.zn(end+1) = mapped_to(to+1);
				if sens_for(to+1)~=-1
					layers{z}.zs = sens_for(to+1);
				end
				if sens_for(from+1)~=-1
					layers{sens_for(from+1)}.znw(end+1) = z;
					if sens_for(to+1)~=-1
						layers{sens_for(from+1)}.zn(end+1) = sens_for(to+1);
					end
				end
				z=z+1;
			end
			
			% weights, biases
			
			for i=1:length(computed)
				if(~isempty(computed{i}))
					compprev{i}=layers{computed{i}}.zp;
					compnext{i}=layers{computed{i}}.zn;
					weight{i}={layers{computed{i}}.zw};
				end
				if(~isempty(sens{i}))
					sensnext{i}=layers{sens{i}}.zn;
					weightout{i}={layers{sens{i}}.znw};
				end
			end
			
			for i=1:length(computed)
				if length(layers{computed{i}}.zn)==0
					outputs(end+1) = computed{i};
					layers{computed{i}}.x_start=0;
					layers{computed{i}}.y_start=0;
					layers{computed{i}}.d_start=0;
				end
			end
			
			if(length(inputs)>1)
				error('Can only have one input');
			end
			if(length(outputs)>1)
				error('Can only have one output');
			end
			if(length(errors)>1)
				error('Can only have one error');
			end
			if(length(labels)>1)
				error('Can only have one label');
			end
			
			for i=1:length(outputs)
				if(sens_forp(outputs(i))~=-1)
					errors = sens_forp(outputs(i));
					labels = layers{sens_forp(outputs(i))}.zl;
				else
					errors = [];
					labels = [];
				end
			end
			
			m2=dn.m;
			m2.layers=layers;
			m2.layer_map=struct('input',{inputs(1)},'computed',{computed},'sens',{sens},'output',{outputs(1)},'error',{errors},'label',{labels},'computed_prev',{compprev},'computed_next',{compnext},'sens_next',{sensnext},'weight',{weight},'bias',{bias},'weight_out',{weightout},'minibatch_index',{1});
		end %save
	
	end %methods

end %class

