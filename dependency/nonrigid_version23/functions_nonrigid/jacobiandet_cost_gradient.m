function [O_error, O_grad]=jacobiandet_cost_gradient(O_grid,sizes,O_goal,sizeI,Spacing)
% Convert Grid vector to grid matrix
O_grid=reshape(O_grid,sizes);

% Delta step used for error gradient
step=0.001;

if(nargout>1)
    if(length(Spacing)==2)
        [O_error1,O_grad1]=jacobiandet_error_2d_double(O_grid(:,:,1),O_grid(:,:,2),sizeI,Spacing(1),Spacing(2));
    else
        [O_error1,O_grad1]=jacobiandet_error_3d_double(O_grid(:,:,:,1),O_grid(:,:,:,2),O_grid(:,:,:,3),sizeI,Spacing(1),Spacing(2),Spacing(3));
    end
else
    if(length(Spacing)==2)
        O_error1=jacobiandet_error_2d_double(O_grid(:,:,1),O_grid(:,:,2),sizeI,Spacing(1),Spacing(2));
    else
        O_error1=jacobiandet_error_3d_double(O_grid(:,:,:,1),O_grid(:,:,:,2),O_grid(:,:,:,3),sizeI,Spacing(1),Spacing(2),Spacing(3));
    end
end

O_org=(O_goal-O_grid).^2;
O_error2=sum(O_org(:))/numel(O_goal);

if(nargout>1)
    if(length(Spacing)==2)
        E_gradpx=(O_goal(:,:,1)-(O_grid(:,:,1)+step)).^2-O_org(:,:,1);
        E_gradpy=(O_goal(:,:,2)-(O_grid(:,:,2)+step)).^2-O_org(:,:,2);
        O_grad2(:,:,1)=E_gradpx/step;
        O_grad2(:,:,2)=E_gradpy/step;
        O_grad2=O_grad2 /numel(O_goal);
    else
        E_gradpx=(O_goal(:,:,:,1)-(O_grid(:,:,:,1)+step)).^2-O_org(:,:,:,1);
        E_gradpy=(O_goal(:,:,:,2)-(O_grid(:,:,:,2)+step)).^2-O_org(:,:,:,2);
        E_gradpz=(O_goal(:,:,:,3)-(O_grid(:,:,:,3)+step)).^2-O_org(:,:,:,3);
        O_grad2(:,:,:,1)=E_gradpx/step;
        O_grad2(:,:,:,2)=E_gradpy/step;
        O_grad2(:,:,:,3)=E_gradpz/step;
        O_grad2=O_grad2 /numel(O_goal);
    end
end

if(nargout>1)
    O_grad=O_grad2 + O_grad1;
end
O_error=O_error2 + O_error1;
