function sendEmailReport(ME,recipients)
if nargin < 2, recipients = 'alejandro@sccn.ucsd.edu';end

%msg = sprintf('%s \n Do you want to email this error to MoBILAB developers?',ME.message);
%[choice,emailSender] = sendReport(msg);
emailSender = '';
% errordlg2(ME.message);
%if strcmp(choice,'Yes')
    message = sprintf('From: %s\n\n%s',emailSender,ME.getReport);
    subject = ['MoBILAB bug: ' ME.message];
    save('ME.mat','ME');
    attachments = 'ME.mat';
    try
        setpref('Internet','SMTP_Server','smtp.ucsd.edu');
        sendmail(recipients,subject,message,attachments)
    end
    delete('ME.mat');
%end