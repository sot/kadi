Search.setIndex({docnames:["api","commands_states","event_descriptions","events","index","maneuver_templates"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["api.rst","commands_states.rst","event_descriptions.rst","events.rst","index.rst","maneuver_templates.rst"],objects:{"kadi.commands":{commands:[0,0,0,"-"],states:[0,0,0,"-"]},"kadi.commands.commands":{CommandTable:[0,1,1,""],get_cmds:[0,3,1,""],get_cmds_from_backstop:[0,3,1,""]},"kadi.commands.commands.CommandTable":{add_cmds:[0,2,1,""],as_list_of_dict:[0,2,1,""],fetch_params:[0,2,1,""]},"kadi.commands.states":{ACISFP_SetPointTransition:[0,1,1,""],ACISTransition:[0,1,1,""],AutoNPMDisableTransition:[0,1,1,""],AutoNPMEnableTransition:[0,1,1,""],BaseTransition:[0,1,1,""],DitherDisableTransition:[0,1,1,""],DitherEnableTransition:[0,1,1,""],DitherParamsTransition:[0,1,1,""],EclipseEntryTimerTransition:[0,1,1,""],EclipsePenumbraEntryTransition:[0,1,1,""],EclipsePenumbraExitTransition:[0,1,1,""],EclipseUmbraEntryTransition:[0,1,1,""],EclipseUmbraExitTransition:[0,1,1,""],EphemerisTransition:[0,1,1,""],EphemerisUpdateTransition:[0,1,1,""],FixedTransition:[0,1,1,""],Format1_Transition:[0,1,1,""],Format2_Transition:[0,1,1,""],Format3_Transition:[0,1,1,""],Format4_Transition:[0,1,1,""],Format5_Transition:[0,1,1,""],Format6_Transition:[0,1,1,""],HETG_INSR_Transition:[0,1,1,""],HETG_RETR_Transition:[0,1,1,""],LETG_INSR_Transition:[0,1,1,""],LETG_RETR_Transition:[0,1,1,""],ManeuverTransition:[0,1,1,""],NMM_Transition:[0,1,1,""],NPM_Transition:[0,1,1,""],NoTransitionsError:[0,4,1,""],NormalSunTransition:[0,1,1,""],ObsidTransition:[0,1,1,""],OrbitPointTransition:[0,1,1,""],ParamTransition:[0,1,1,""],RadmonDisableTransition:[0,1,1,""],RadmonEnableTransition:[0,1,1,""],SCS84DisableTransition:[0,1,1,""],SCS84EnableTransition:[0,1,1,""],SCS98DisableTransition:[0,1,1,""],SCS98EnableTransition:[0,1,1,""],SPMDisableTransition:[0,1,1,""],SPMEclipseEnableTransition:[0,1,1,""],SPMEnableTransition:[0,1,1,""],SimFocusTransition:[0,1,1,""],SimTscTransition:[0,1,1,""],StateDict:[0,1,1,""],SubFormatEPS_Transition:[0,1,1,""],SubFormatNRM_Transition:[0,1,1,""],SubFormatPDG_Transition:[0,1,1,""],SubFormatSSR_Transition:[0,1,1,""],SunVectorTransition:[0,1,1,""],TargQuatTransition:[0,1,1,""],TransKeysSet:[0,1,1,""],TransitionMeta:[0,1,1,""],add_transition:[0,3,1,""],decode_power:[0,3,1,""],get_chandra_states:[0,3,1,""],get_continuity:[0,3,1,""],get_states:[0,3,1,""],get_transition_classes:[0,3,1,""],get_transitions_list:[0,3,1,""],interpolate_states:[0,3,1,""],print_state_keys_transition_classes_docs:[0,3,1,""],reduce_states:[0,3,1,""]},"kadi.commands.states.ACISFP_SetPointTransition":{set_transitions:[0,2,1,""]},"kadi.commands.states.ACISTransition":{set_transitions:[0,2,1,""]},"kadi.commands.states.BaseTransition":{get_state_changing_commands:[0,2,1,""]},"kadi.commands.states.DitherParamsTransition":{set_transitions:[0,2,1,""]},"kadi.commands.states.EphemerisUpdateTransition":{set_transitions:[0,2,1,""]},"kadi.commands.states.FixedTransition":{set_transitions:[0,2,1,""]},"kadi.commands.states.ManeuverTransition":{add_manvr_transitions:[0,2,1,""],callback:[0,2,1,""]},"kadi.commands.states.NormalSunTransition":{callback:[0,2,1,""]},"kadi.commands.states.ParamTransition":{set_transitions:[0,2,1,""]},"kadi.commands.states.SPMEclipseEnableTransition":{set_transitions:[0,2,1,""]},"kadi.commands.states.StateDict":{copy:[0,2,1,""]},"kadi.commands.states.SunVectorTransition":{set_transitions:[0,2,1,""],update_sun_vector_state:[0,2,1,""]},"kadi.commands.states.TargQuatTransition":{set_transitions:[0,2,1,""]},"kadi.events":{models:[0,0,0,"-"],query:[0,0,0,"-"]},"kadi.events.models":{AsciiTableEvent:[0,1,1,""],BaseEvent:[0,1,1,""],BaseModel:[0,1,1,""],CAP:[0,1,1,""],DarkCal:[0,1,1,""],DarkCalReplica:[0,1,1,""],DsnComm:[0,1,1,""],Dump:[0,1,1,""],Dwell:[0,1,1,""],Eclipse:[0,1,1,""],Event:[0,1,1,""],FaMove:[0,1,1,""],GratingMove:[0,1,1,""],IFotEvent:[0,1,1,""],IntervalPad:[0,1,1,""],LoadSegment:[0,1,1,""],LttBad:[0,1,1,""],MajorEvent:[0,1,1,""],Manvr:[0,1,1,""],ManvrSeq:[0,1,1,""],MyManager:[0,1,1,""],NormalSun:[0,1,1,""],Obsid:[0,1,1,""],Orbit:[0,1,1,""],OrbitPoint:[0,1,1,""],Pad:[0,1,1,""],PassPlan:[0,1,1,""],RadZone:[0,1,1,""],SafeSun:[0,1,1,""],Scs107:[0,1,1,""],TlmEvent:[0,1,1,""],TscMove:[0,1,1,""],Update:[0,1,1,""],fuzz_states:[0,3,1,""],get_event_models:[0,3,1,""],import_ska:[0,3,1,""],msidset_interpolate:[0,3,1,""]},"kadi.events.models.AsciiTableEvent":{get_events:[0,2,1,""],get_extras:[0,2,1,""]},"kadi.events.models.BaseEvent":{get_commands:[0,2,1,""],get_next:[0,2,1,""],get_previous:[0,2,1,""]},"kadi.events.models.BaseModel":{QuerySet:[0,1,1,""],from_dict:[0,2,1,""],get_model_fields:[0,2,1,""],get_obsid:[0,2,1,""]},"kadi.events.models.BaseModel.QuerySet":{select_overlapping:[0,2,1,""],table:[0,2,1,""]},"kadi.events.models.CAP":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.DarkCal":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.DarkCalReplica":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.DsnComm":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_extras:[0,2,1,""]},"kadi.events.models.Dump":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.Dwell":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.Eclipse":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.FaMove":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_extras:[0,2,1,""]},"kadi.events.models.GratingMove":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_extras:[0,2,1,""],get_state_times_bools:[0,2,1,""]},"kadi.events.models.IFotEvent":{get_events:[0,2,1,""]},"kadi.events.models.LoadSegment":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.LttBad":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_extras:[0,2,1,""]},"kadi.events.models.MajorEvent":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_events:[0,2,1,""]},"kadi.events.models.Manvr":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_dwells:[0,2,1,""],get_events:[0,2,1,""],get_manvr_attrs:[0,2,1,""],get_one_shot:[0,2,1,""],get_target_attitudes:[0,2,1,""]},"kadi.events.models.ManvrSeq":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.NormalSun":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.Obsid":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_events:[0,2,1,""]},"kadi.events.models.Orbit":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_events:[0,2,1,""]},"kadi.events.models.OrbitPoint":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.PassPlan":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.RadZone":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.models.SafeSun":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_state_times_bools:[0,2,1,""]},"kadi.events.models.Scs107":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_state_times_bools:[0,2,1,""]},"kadi.events.models.TlmEvent":{fetch_event:[0,2,1,""],get_events:[0,2,1,""],get_extras:[0,2,1,""],get_msids_states:[0,2,1,""],get_state_times_bools:[0,2,1,""],msidset:[0,2,1,""],plot:[0,2,1,""]},"kadi.events.models.TscMove":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""],get_extras:[0,2,1,""]},"kadi.events.models.Update":{DoesNotExist:[0,4,1,""],MultipleObjectsReturned:[0,4,1,""]},"kadi.events.query":{EventQuery:[0,1,1,""],get_dates_vals:[0,3,1,""]},"kadi.events.query.EventQuery":{all:[0,2,1,""],filter:[0,2,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:exception"},terms:{"01t00":3,"03722364e":1,"04030078e":1,"09485195e":1,"0x309e8d0":3,"0x358f5d0":3,"0x7fa4c1492790":0,"0x7fa4c154b100":0,"0x7fa4c154b4c0":0,"0x7fa4c154b6d0":0,"0x7fa4c154b7c0":0,"0x7fa4c154b910":0,"0x7fa4c154bd60":0,"0x7fa4c154d340":0,"0x7fa4c15c2e50":0,"0x7fa4c15c2f40":0,"0x7fa4c15c2fa0":0,"0x7fa4c62823d0":0,"11564724e":1,"14710906e":1,"15954877e":1,"16664564e":1,"17512532e":1,"1dpamzt":1,"23403971e":1,"24055031e":1,"27899874e":1,"28324009e":1,"33533892e":1,"38460120e":1,"3famov":[0,2],"3fapo":3,"3mrmmxmv":[0,2],"3rd":1,"3sdagv":3,"3sdfatsv":3,"3sdm15v":3,"3sdp15v":3,"3sdp5v":3,"3sdtstsv":3,"3tscmove":[0,2],"40000915e":1,"45077701e":1,"4hposaro":3,"4mp28av":[0,2],"4ohetgin":[0,1],"4ohetgr":0,"4oletgin":0,"4oletgr":0,"51367966e":1,"57368959e":1,"5ehse300":3,"6th":1,"76678196e":1,"78304550e":1,"7c063c0":1,"83613678e":1,"86236582e":1,"90427812e":1,"92042461e":1,"boolean":[0,2],"break":[0,1,3],"case":[0,1,2,3],"char":[0,2,3],"class":[0,1,2,3,5],"default":[0,1,3],"final":[0,1,3],"float":[0,2,3],"function":[0,1,3],"import":[0,1,3],"long":[0,2,3],"new":[0,1,3],"return":[0,1,3],"short":[0,2],"static":0,"true":[0,1,2,3],"try":[0,1],"while":[0,1,3],AND:3,But:3,EPS:0,For:[0,1,2,3],NOT:3,Not:1,One:[0,1,2,3],POS:0,SCS:[0,1,2,4],That:1,The:[0,1,2,3,4,5],There:[1,3],These:[0,1,2,3,5],Use:4,Using:[1,3],With:[1,3],__dates__:[0,1],__exact:[0,3],__init__:3,__repr__:0,aa00000000:1,aacccdpt:3,abiasz:3,about:[1,3],abov:[0,1,2,3],aca:[0,1,3],aca_proc_act_start:[0,2,3],accept:3,access:[1,3,4],accordingli:[0,1],account:1,accumul:1,accur:[0,2],aci:[0,1,2,3],acisfp_setpoint:[0,1],acisfp_setpointtransit:[0,1],acispkt:[0,1],acistransit:[0,1],acq:[0,2],acq_start:[0,2],acquisit:[0,2],act:[0,2],activ:[0,2,3],actual:[0,1,2,3],adac:0,adapt:1,add:[0,1],add_cmd:0,add_command:1,add_manvr_transit:0,add_transit:0,adding:0,addit:[0,2,3],adjac:[0,1],administr:4,advantag:3,affect:[1,3],aflcrset:0,after:[0,1,2,3],afterward:3,again:[1,3],airu2bt:3,ala:0,all:[0,1,3],allow:[0,1,3,4],allow_parti:[0,3],alon:1,along:[0,1,5],alreadi:1,also:[0,1,2,3],although:[1,3],alwai:[0,1,3],analysi:[3,4],angl:[0,2,3],angle__gt:[0,3],ani:[0,1,2,3],anomal:[0,2],anomali:[1,4,5],anoth:[0,3],answer:1,anyth:[0,2],ao1minu:[0,1],ao1plu:[0,1],aoacaseq:[0,2],aoacaseq_aqxn_brit:5,aoacaseq_aqxn_guid:5,aoacaseq_brit_kalm:5,aoacaseq_guid_kalm:5,aoacaseq_kalm_aqxn:5,aoacrstd:1,aoargper:[0,1],aoascend:[0,1],aoatter1:0,aoattqt1:3,aoditpar:0,aodsdith:0,aoeccent:[0,1],aoeclip:[0,2,3],aoendith:0,aoephem1:[0,1],aoephem2:[0,1],aoephup:0,aofattmd:[0,2],aofattmd_mnvr_stdi:5,aofattmd_null_stdi:5,aofattmd_stdy_mnvr:5,aofattmd_stdy_nul:5,aofuncd:[0,1],aofuncen:0,aoiterat:[0,1],aomanuvr:0,aomot:[0,1],aonm2np:[0,1],aonm2npd:0,aonmmod:[0,1],aonpmod:0,aonsmsaf:0,aoorbang:[0,1],aopcadmd:[0,2],aopcadmd_nman_npnt:5,aopcadmd_nman_nsun:5,aopcadmd_npnt_nman:5,aopcadmd_nsun_nman:5,aopcads:0,aopcadsd:[0,1],aoperig:[0,1],aopsacpr:[0,2],aoratio:[0,1],aorwbia:[0,2],aosini:[0,1],aoslr:[0,1],aosqrtmu:[0,1],aostrcat:1,aounload:[0,2],apoge:[0,2,3],app:0,appear:[0,2],appli:[0,1,3],applic:1,appropri:[0,3],approv:[0,1],aqxn:[0,2],archiv:3,arcsec:[0,2],arg:[0,1,3],argument:[0,1,3],arguments:3,around:3,arrai:0,as_list_of_dict:0,ascend:[0,2,3],ascens:[0,2],ascii:1,asciitableev:0,ask:3,aspect:3,assembl:[0,1,3],assist:[0,1],associ:[0,1,2,3],assum:0,astropi:[0,1,3],astut:1,attempt:[0,2],attitud:[0,1,2,3],attribut:[0,1,2,3,5],auto_npnt:[0,1],automat:[0,1],autonom:[1,4],autonpmdisabletransit:[0,1],autonpmenabletransit:[0,1],aux_msidset:0,avail:[0,1,2,3],awai:3,awar:3,back:[0,1],backstop:[0,1],backstop_fil:1,bad:0,baffin:3,base:[0,1,3,4],baseclass:0,baseev:0,basemodel:0,basetransit:[0,1],basi:0,basic:[1,3],batteri:0,becaus:[1,3],becom:1,been:[1,3],befor:[0,1,2,3],begin:[0,2,3],below:[1,3],beteween:0,between:[0,1,2,3],bigger:[0,3],bit:[1,3],board:[0,1,2,4],bool:0,bot:[0,2,3],both:[0,3],bound:3,brand:3,bright:[3,4],brows:4,bs_cmd:1,built:1,bump:[0,2],bunch:3,c1sqax:[0,2],cal:[3,4],calibr:[0,3],call:[0,1],callback:[0,1],came:3,can:[0,1,2,3],canon:0,cap:[0,3,4],capabl:4,captur:1,care:1,catalog:[0,1,3],caus:1,ccd:0,ccd_count:[0,1],certain:1,cfa:[0,2,3],chain:3,challeng:1,chandra:[0,2],chandra_major_ev:[0,2],chang:[0,1,2],charact:3,character:[0,2],characterist:3,check:[0,1],chip:0,chop:3,cimodesl:1,ciu1024t:1,ciu1024x:1,ciu512t:1,ciumacac:[0,2],classic:[0,1],classifi:[0,2],classmethod:0,claus:[0,3],clip:3,clock:[0,1],close:3,cls:[0,1],cmd:[0,1,3],cmd_attr:0,cmd_count:[0,2],cmd_param_kei:[0,1],cmd_state:1,cmdlist:0,cmds_type:1,coaosqid:1,cobrqid:0,cobsrqid:[0,2],code:1,codisas1:0,codisasx:0,coenas1:0,coenasx:0,collect:[0,1],column:[0,1,3],com:[0,3],combin:[0,1,2],come:[1,3],comm:[0,3],comma:1,command:3,command_attribut:[0,1],command_hw:[0,1],command_param:0,command_sw:[0,1],commandt:[0,1],comment:[0,2,3],common:[0,2,3],compact:[0,1],comparison:[0,3],compart:0,compat:[0,1,3],complet:[0,1,3],complex:[0,1,3],complic:3,composit:[0,3],comprehens:1,compris:[0,2],comput:[0,1,3],concept:1,condit:[0,2],config:[0,2,3],configur:[0,1,2,3],conlofp:[0,2],connect:0,consid:[0,1,2],consist:4,consol:1,construct:3,contain:[0,1,2,3],content:3,contigu:[0,2],continu:[0,4],control:[0,1,2],conveni:1,convent:3,convers:[1,3],convert:[0,2,3],copi:[0,1,3],copy_indic:0,coradmen:[0,2],correspond:[0,1,2,3],could:[0,1,3],count:[0,2],coupl:1,cours:1,cover:[0,2,3],crazi:3,creat:[1,3],crew:[0,2],criteria:[0,3],cross:[0,2,3],cselfmt1:0,cselfmt2:0,cselfmt3:0,cselfmt4:0,cselfmt5:0,cselfmt6:0,csh:3,cti:1,ctu:[0,2],ctufmtsl:[0,2],current:[0,1,3],curv:0,custom:[0,1],cut:3,cxc:[0,2,3],d80000300030603001300:1,dai:[0,1,2,3],daili:3,dark:[0,3,4],dark_cal:3,dark_cal_replica:3,darkcal:[0,3],darkcalreplica:[0,3],dat:3,dat_good:3,data:[0,2,3],data_dir:0,data_r:[0,2,3],databas:[0,1,3,4],dataset:[0,3],date:[0,2,3],datestart:[0,1],datestop:[0,1],datetim:[0,1,3],ddd:[0,2],debug:0,dec:[0,1,2],declin:[0,2],decod:0,decode_pow:0,decor:0,def:1,default_state_kei:[0,1],default_valu:0,defin:[0,2,3],definit:[0,2],defint:3,deg:[0,2],degre:[0,2],delimit:1,demo:3,demonstr:3,depend:0,deriv:[0,3,5],descr:[0,2,3],describ:[0,3],descript:[0,3,4],descriptor:0,design:1,desir:[0,1],detail:[0,1,2,4],detect:[0,2],detector:[0,2],determin:[0,1],dict:[0,1],dictionari:[0,1],did:[1,3],didn:3,differ:[0,1,2,3],dig:1,direct:[0,2,3],directli:[0,1,3],directori:1,disa:[0,2],disabl:[0,2,3],discourag:3,discret:1,discuss:3,disturb:3,dither:[0,1],dither_ampl_pitch:[0,1],dither_ampl_yaw:[0,1],dither_period_pitch:[0,1],dither_period_yaw:[0,1],dither_phase_pitch:[0,1],dither_phase_yaw:[0,1],ditherdisabletransit:[0,1],ditherenabletransit:[0,1],ditherparamstransit:[0,1],django:[0,3,4],djangoproject:[0,3],doc:[0,3],docstr:[1,3],document:[1,3,4],doe:[0,1,2,3],doesnotexist:0,doi:[0,2,3],doing:[1,3],don:1,done:1,doubl:[0,3],down:[0,3],download:3,downstream:[0,1],dp_:3,dramat:1,drive:1,dry:0,dsn:[0,3,4],dsn_comm:3,dsncomm:[0,3],dt_start_radzon:[0,2,3],dt_stop_radzon:[0,2,3],dtype:0,due:[0,1,2],dump:[0,3,4],dur:[0,2,3],durat:[0,1,2,3],dure:[0,1,2,3],dwell:[0,3,4],dwell_set:3,dynam:[0,1,4],e1300:3,e1300_log:3,e1300_rad_zon:3,each:[0,1,2,3],earli:[0,1,2],earlier:[0,1,3],easi:[1,3,4],easiest:[1,3],easili:3,ecl:[0,2,3],eclips:[0,1,3],eclipse_tim:[0,1],eclipseentrytimertransit:[0,1],eclipsepenumbraentrytransit:[0,1],eclipsepenumbraexittransit:[0,1],eclipseumbraentrytransit:[0,1],eclipseumbraexittransit:[0,1],edu:[0,2,3],effect:1,effort:1,either:[0,1,3,4],element:[1,4],elif:1,embed:0,enab:[0,2],enabl:[0,2],encapsul:[1,3],end:[0,2,3,5],endswith:[0,3],eng:[0,2],engarch:[0,3],engin:[0,2,3,4],enough:1,enter:3,entir:[0,1,3],entri:[0,1,2],environ:3,eodai:0,eoecleto:0,eoestecn:0,eonight:0,eot:[0,2,3],ephem_upd:[0,1],ephemeri:0,ephemeristransit:[0,1],ephemerisupdatetransit:[0,1],ephin:3,eqf013m:1,equal:[0,3],equival:1,err_log:[0,2],error:[0,2],essenti:3,est_datetim:[0,2],establish:3,estim:1,etc:[1,3,4],evalu:1,even:[0,1,3],event:[1,5],event_msid:0,event_msidset:0,event_typ:[0,1],event_v:0,eventqueri:[0,3],everi:[0,1,3,4],everyth:3,exact:[0,1,3],exactli:[0,1,2,3],examin:3,exampl:[0,1,3,5],except:[0,3],exclud:3,execut:0,exit:[0,1],expect:[0,3],explain:3,express:[0,3],extra:0,extra_msid:0,fa_mov:3,fact:[1,3],factor:[0,1],fairli:1,fals:[0,1,3],famov:[0,3],fdb:[0,2,3],fdb_web:[0,2],featur:[0,1,3],fep:0,fep_count:[0,1],fetch:[0,1,3],fetch_ev:0,fetch_event_msid:0,fetch_msid:0,fetch_param:[0,1],fetch_stat:1,few:[0,1,3],field:[0,2,3],field_nam:[0,3],fig:0,figsiz:0,figur:3,file:[0,1,3],filenam:0,filter:[0,1,4],filter_bad:0,filter_kwarg:0,filter_typ:[0,3],find:[0,1,3],finish:3,fire:3,first:[0,1,2,3],fit:3,fix:[0,1],fixedtransit:[0,1],flag:[0,2,3],flexibl:1,fmt1:0,fmt2:0,fmt3:0,fmt4:0,fmt5:[0,2],fmt6:0,focal:0,focu:[0,3],follow:[0,1,2,3,4,5],fontsiz:3,forc:0,foreignkei:[0,2],form:[0,3],format1_transit:0,format2_transit:0,format3_transit:0,format4_transit:0,format5_transit:0,format6_transit:0,format:[0,1,2,3],fot:[0,2,3],fot_web:[0,2],found:0,four:1,framework:4,frequent:[1,3],friendli:0,from:[0,3,4],from_dict:0,fsw:[0,2,3],full:[0,2,3],fully_radzone_manvr:0,func:0,fundament:1,further:[0,3],fuzz:0,fuzz_stat:0,fuzzed_st:0,gap:1,gcm_tscacc:3,gener:[0,1,2,3,4],get:[0,1,4],get_chandra_st:[0,1],get_cmd:[0,1],get_cmds_from_backstop:0,get_command:0,get_continu:[0,1],get_dates_v:[0,3],get_dwel:0,get_ev:0,get_event_model:0,get_extra:0,get_manvr_attr:0,get_model_field:0,get_msids_st:0,get_next:0,get_obsid:[0,3],get_one_shot:0,get_previ:0,get_stat:[0,1],get_state_changing_command:[0,1],get_state_times_bool:0,get_target_attitud:0,get_time_rang:1,get_transition_class:0,get_transitions_list:0,git:3,give:[0,2],given:[0,1,3],global:0,going:[1,3],good:3,good_tim:3,got:[0,2],grate:[0,1,3],grating_mov:3,gratingmov:[0,3],greta:[3,4],grid:3,grnd:[0,2],ground:[0,1,3,4],group:1,gte:[0,3],guid:[0,2],guide_start:[0,2],had:[1,3],hand:[0,2],handi:[0,1],handl:[0,1,3],happen:[1,3],happi:1,harvard:[0,2,3],has:[0,1,2,3],have:[0,1,2,3],hdf5:1,head:[3,4],help:[1,4],here:[0,1,3],hetg:[0,1,3],hetg_insert:3,hetg_insr_transit:[0,1],hetg_retr_transit:[0,1],hex:1,high:[0,3],highli:1,hint:0,histori:[0,1],hold:[3,4],hook:0,host:3,hour:[0,1,2],how:[1,3],howev:[0,1,2,3],hrc:[0,2,3],htm:[0,2],html:[0,2],http:[0,2,3],icxc:4,idea:[1,3],ident:[0,1],identif:1,identifi:[0,1,3],idx:[0,1],ifot:[0,1,3],ifot_id:[0,2,3],ifotev:0,ignor:[0,2,3],impact:1,impati:3,implement:[0,3],import_ska:0,inac:[0,2],includ:[0,1,2,3],inclus:3,inclusive_stop:0,incomplet:3,increas:1,index:[0,1],indic:[0,1,2,3],individu:[0,2],inform:[0,1,2,3],ingest:1,inherit:1,init:[0,2],initi:0,input:[0,1],insensit:0,insert:[0,1,3],insid:3,insist:3,insr:[0,1,2,3],instal:1,instanc:[0,1,3,5],instancemethod:3,instanti:1,instead:[0,1,2],instrument:[0,2],int64:1,integ:[0,2,3],integr:4,interact:3,interest:[0,1,3],interfac:[0,3],interm_att:3,intermedi:[0,3],intern:[0,1],interpol:0,interpolate_st:0,interrupt:1,interv:[0,1,4],interval_pad:[0,3],intervalpad:0,ipython:[1,3],isnul:[0,3],item:[1,3],iu_mode_select:1,iumodeselecttransit:1,jan0818:1,jan:[1,3],join:[0,2],just:[0,1,3],kadi:[0,1,2,3],kalm:[0,2],kalman:[0,3],kalman_start:[0,2],kei:[0,2,3],keyword:[0,1,3],know:1,known:0,ksec:[0,3],kwarg:[0,3],label:3,larger:[0,2],last:[0,1,3],last_tlm_dat:1,later:[0,3],latter:0,launch:[3,4],lazi:[0,1],learn:3,least:[0,3],leav:3,left:0,legaci:1,legend:3,length:1,less:[0,2],let:[1,3],letg:[0,1,3],letg_insr_transit:[0,1],letg_retr_transit:[0,1],level:[0,3],lga:[0,2],like:[0,1,3],likewis:[0,3],limit:3,line:0,link:[0,2,4],list:[0,1,2,3],live:0,load:[0,1,3,4],load_nam:[0,2],load_seg:3,load_start:1,loadseg:[0,3],local:[0,2],lock:3,log10:3,log:[0,2],logger:0,logic:[0,1,3],logical_interv:0,longer:[0,3],look:[0,1,3],lookback:[0,1],lookup:[0,3],loop:1,lot:3,lspentri:0,lspexit:0,lte:[0,3],ltt:0,ltt_bad:3,lttbad:[0,3],mai:[0,1,2,3],main:[0,1],main_arg:0,maintain:[0,1,2,4],major:[0,3,4],major_ev:[0,2,3],majorev:[0,2,3],make:[0,1,3],manag:0,maneuv:[0,1,3,4],maneuvertransit:[0,1],mani:[0,1,3],manipul:[3,4],manual:1,manuev:3,manv:[0,2],manveuv:3,manvr:[0,2,3,5],manvr_seq:3,manvr_start:[0,2],manvr_stop:[0,2],manvr_templ:3,manvrs_2011:3,manvrseq:[0,3],march:[3,5],mask:0,match:[0,1,2,3],match_valu:0,matplotlib:[0,3],maud:1,max:[0,2],max_pwm:[0,2,3],maximum:[0,2,3],mce:[0,2],mean:3,meaning:3,mech:[3,4],mechan:[0,1],member:0,merg:[0,1,2],merge_ident:[0,1],messag:1,meta:0,metaclass:0,method:[0,1,2,3],mid:1,might:[0,3],millisec:1,minim:1,minut:[0,1,3],miss:[0,1,3],mission:[0,1,2,4,5],mnem:0,mnvr:[0,2],mode:[0,1,3,4],model:[1,2,3],model_dict:0,model_nam:0,modelclass:0,modul:[0,1,3],moment:[1,3],momentum:[0,3,4],mon:[0,2,3],monitor:[0,2],more:[0,1,2,4],most:[0,1,2,3],mostli:0,motion:[0,2,3],move:[0,2,3],movement:[0,3,4],mp_obsid:[0,1],mp_targquat:0,msid:[0,1,2,3],msidset:0,msidset_interpol:0,mta:3,multipl:[0,1,2],multipleobjectsreturn:0,must:[0,1,3],mymanag:0,n_acq:[0,2],n_dwell:[0,2,3],n_dwell__exact:3,n_dwell__lt:[0,3],n_guid:[0,2],n_kalman:[0,2],name:[0,1,2,3],natur:[0,2],ndarrai:0,nearest:[0,2,3],necessari:0,need:[0,1,3],network:[3,4],never:1,nevertheless:1,next:[0,1,2,3],next_manvr_start:[0,2],next_nman_start:[0,2],nman:[0,2],nman_dwel:3,nman_start:[0,2],nmm_transit:[0,1],node:[0,1,2,3],nomin:[0,2],non:[1,4],non_load_cmd:1,non_rad_manvr:0,none:[0,1,3],norm:0,normal:[0,1,3],normal_sun:3,normalsun:[0,2,3],normalsuntransit:[0,1],note:[0,2,3],notebook:3,notic:[0,1],notransitionserror:0,now:[0,1,3],npm:[0,1,3,4],npm_transit:[0,1],npnt:[0,2],npnt_start:[0,2],npnt_stop:[0,2],nrm:0,nrml:[0,2],nsun:[0,2],num:[0,2],number:[0,1,2,3,5],numpi:[0,1],obc:[0,2,3],object:[0,1,3],observ:[0,3],observatori:[3,4],obsid:[0,1,2,4],obsidtransit:[0,1],occur:[0,1,3,5],occurr:5,occweb:[0,2],off:[0,2],off_nom_rol:[0,1],off_nomin:0,off_nominal_rol:0,ofl:1,ofmtsep:0,ofmtsnrm:0,ofmtspdg:0,ofmtsssr:0,ofp:[0,2],often:[0,1],omit:1,onc:[0,1],one:[0,1,2,3],one_shot:[0,2],one_shot_pitch:0,one_shot_rol:0,one_shot_yaw:0,onetoon:[0,2],onli:[0,1,2,3],onto:[0,2,3],oormpd:0,oormpen:0,open:1,oper:3,operation:1,option:[0,1,2,3],orbit:[0,3,4],orbit_num:[0,2,3],orbit_point:[0,1,3],orbitpoint:[0,3],orbitpointtransit:[0,1],orbpoint:[0,1],order:[0,1,3],origin:[1,3],other:[0,1,2,3],ouput:1,our:1,out:[0,3],outfil:1,output:[0,1,3],outsid:3,over:[0,2,3],overlap:[0,1],overplot:3,overrid:0,overridden:0,overview:4,own:1,packet:1,pad:[0,4],pair:[0,1,3],param:[0,1,3],paramet:[0,1],paramtransit:[0,1],parent:0,parse_cm:[0,1],parsecm:0,part:[0,1,2,3],partial:0,particular:[0,1,3],particularli:1,pass:[0,3,4],pass_plan:[0,2,3],passplan:[0,3],pcad:[0,1,2],pcad_mod:[0,1],pdg:0,pentri:0,penumbra:0,per:[0,1],perform:[0,1,3],perige:[0,2,3],period:[0,3],person:1,pexit:0,pitch:[0,1],pitch_stab_perf:3,place:0,plan:[0,1,3],plane:0,pleas:3,plenti:3,plot:[0,3],plot_cxctim:3,plt:3,point:[0,1,3],portion:3,pos:[0,1],posit:[0,1,2,3],possibl:[0,1,3],potenti:[0,1],power:[0,1,2,3],power_cmd:[0,1],practic:[1,3],pre:[0,1,2],precis:[1,3],predict:1,prematur:[0,2,3],press:3,prev_dat:[0,2],prev_manvr_stop:[0,2],prev_npnt_start:[0,2],prev_tim:[0,2],prev_val:[0,2],previou:[0,1,2,3],primari:0,print:[0,1,3],print_state_keys_transition_classes_doc:0,printout:3,prior:[0,1],probabl:1,procedur:1,process:[0,1],product:[1,4],prompt:3,propag:1,properti:0,prove:1,provid:[0,1,2,3,4],pseudo:1,puls:3,pure:0,purpos:0,pwm:[0,2],pylab:3,python:[0,1,3,4],quantiti:1,quasi:0,quaternion:0,queri:[1,3,4],query_ev:0,queryev:[0,3],queryset:[0,3],question:0,quickli:1,quiet:3,rad:[0,2,3],rad_manvr:0,rad_zon:[0,3],radiat:[0,3],radmon:[0,1,2,3],radmondisabletransit:[0,1],radmonenabletransit:[0,1],radzon:[0,2,3],rai:[3,4],ran:1,rang:[0,3],rate:[0,2,3],rather:3,read_backstop:[0,1],reader:1,real:[1,3,4],realli:3,reason:[1,3],recal:1,recarrai:0,record:[0,2],recoveri:[0,1,2,4],red:3,reduc:[0,1],reduce_st:0,redund:0,ref:[0,3],refer:[1,3],reflect:[0,1],regardless:1,regist:1,regular:3,rel:[0,2,3,5],rel_tstart:[0,2],relat:[0,1,2,3],relev:[0,1,2],reli:1,reliabl:[0,2],rememb:3,remind:3,remov:[0,3],remove_interv:[0,3],remove_starcat:0,replan:1,replic:1,replica:[0,3],report:[0,2],repres:[0,1,2,3,5],reproduc:1,req:0,request:1,requir:[0,1,3],resolut:[0,2],respect:[0,1,2],respons:1,result:[0,1,3,4],retain:3,retr:[0,2],retract:0,retriev:1,review:1,right:[0,2,5],roll:[0,1,2],roll_bias_diff:3,roughli:1,routin:0,row:[0,1],rsl:[0,2],rule:[0,2],run:[0,1,3,4],rxa_rsl:[0,2],rxb_rsl:[0,2],safe:[0,3,4],safe_sun:[0,3],safesun:[0,3],sai:[1,3],same:[0,1,3],sampl:[0,2],samytemdel:3,san:1,sapytemdel:3,sce1300:3,sched_support_tim:[0,2],schedul:0,scienc:[0,1,2],scrape:[0,2],screen:3,script:1,scs107:[0,3,4],scs84:[0,1],scs84disabletransit:[0,1],scs84enabletransit:[0,1],scs98:[0,1],scs98disabletransit:[0,1],scs98enabletransit:[0,1],scs:[0,1,2],search:[0,3],sec:[0,2,3],second:[0,1,2,3,5],section:[1,3,5],see:[0,1,2,3],seen:[0,2],segment:[0,1,3],select:[0,1,4],select_interv:[0,3],select_overlap:[0,3],self:[0,3],sens:3,separ:[0,1],sequenc:[0,1,3,4],serv:1,set:[0,1,3],set_transit:[0,1],setpoint:0,share:3,shell:3,shortcut:[0,3],shot:[0,2],should:[0,1],show:[0,1,2,3,5],shown:[1,3],si_mod:[0,1],side:[0,1,3],signatur:1,sim:[0,1,3],simfa_po:[0,1],simfocu:0,simfocustransit:[0,1],similar:[0,3],simpl:[0,1,3],simpo:[0,1],simtran:[0,1],simtsctransit:[0,1],sinc:[0,1,3,4],singl:[0,1,2,3],site:[0,2,3,4],situat:3,size:1,ska:[0,1,3,4],skatest:3,skip:[0,3],slice:[0,3],slot:[0,2],small:3,snap:3,snip:3,soe:[0,2,3],softwar:[0,2],some:[0,1,3],someth:1,sometim:[1,3],sort:[0,1,3],sot:1,sourc:[0,1,2,3,4],space:1,spacecraft:[0,1,2,3],span:[0,2,3],speak:1,special:[0,1,3],specif:[0,1,2,3],specifi:[0,1,3],spmdisabletransit:[0,1],spmeclipseenabletransit:[0,1],spmenabletransit:[0,1],sql:[0,3],ssr:0,sss:3,stabl:3,stackoverflow:0,standard:3,star:[0,1,2,3,4],start:[0,1,2,4],start_3fapo:[0,2],start_3tscpo:[0,2,3],start_4hposaro:[0,2],start_4lposaro:[0,2],start_dec:[0,2],start_det:[0,2,3],start_ra:[0,2],start_radzon:[0,2,3],start_rol:[0,2],startswith:[0,3],stat:3,state:2,state_interv:0,state_kei:[0,1],statedict:0,statement:3,station:[0,2,3],stdout:[0,1],stdy:[0,2],step:[0,1,2,3],stop:[0,1,2,3],stop_3fapo:[0,2],stop_3tscpo:[0,2,3],stop_4hposaro:[0,2],stop_4lposaro:[0,2],stop_aotarqt:0,stop_dec:[0,2],stop_det:[0,2,3],stop_ra:[0,2],stop_radzon:[0,2,3],stop_rol:[0,2],stoppag:1,store:[0,1,3],str21:1,str8:1,str:[0,1],straightforward:1,strategi:3,strictli:1,string:[0,1,3],structur:[0,2],sts:1,stuff:0,sub:1,subclass:0,subformat:0,subformateps_transit:0,subformatnrm_transit:0,subformatpdg_transit:0,subformatssr_transit:0,subsequ:[0,3],subset:[0,3],subtleti:[1,3],suffici:0,sun:[0,1,3],sun_pos_mon:[0,1],sunvectortransit:[0,1],supplement:[0,2],suppli:[0,1,3],support:[0,1,2],sure:1,syntax:3,system:[0,1,2],t_fuzz:0,t_perige:[0,2,3],tab:3,tabl:[0,1,2,4],tabular:0,tail:1,take:3,targ_q1:[0,1],targ_q2:[0,1],targ_q3:[0,1],targ_q4:[0,1],target:0,targquattransit:[0,1],te_00a02:1,team:1,telem:1,telemet:0,telemetri:[0,1,2,4],tell:1,temperatur:0,templat:[0,2,3,4],term:3,termin:3,test:[0,1,3],test_stat:1,text:[0,2,3],tfcag:3,tfcdg:3,than:[0,2,3],thei:[0,1,3],them:[1,3],themselv:3,thermal:1,thi:[0,1,2,3,4],thie:3,thing:1,those:[0,1,3,4],though:[1,3],three:3,three_acq:3,through:[0,1],time0:0,time:[0,1,2,4,5],timecnt:0,timelin:0,timeline_id:1,timer:0,titl:[0,2,3],tlmevent:0,tlmsid:[0,1],togeth:[0,2,3],toggl:[0,2],tom:3,tool:4,tool_doc:3,top:3,topic:[0,3],total:[0,1,3],track:[0,2,3],trans_kei:[0,1],transit:[0,1,2],transition_kei:[0,1],transition_v:[0,1],transitionmeta:0,transitions_class:0,transitions_dict:0,transkeysset:0,translat:[0,3],trend:3,trickeri:1,tsc:[0,3],tsc_move:[0,3],tscmove:[0,3],tstart:[0,2,3],tstop:[0,2,3],tupl:0,turn:3,two:[0,1,2,3],two_acq:3,txt:1,type:[0,1,2,3],typic:[0,1,2],umbra:0,unaffect:1,unchang:[0,1,2],underli:[0,3],underscor:[0,3],understand:3,unfilt:0,uniqu:[0,1,2,3],unkn:[0,2],unknown:[0,2,3],unlik:[0,2],until:1,untouch:3,updat:[0,1,2],update_ev:0,update_sun_vector_st:0,usag:[1,3],use:[1,3],used:[0,1,3,5],useful:[0,1,3],user:0,uses:4,using:[0,1,3,4],usual:[0,1,3],val:[0,2,3],valid:1,valu:[0,1,2,3],valueerror:0,varieti:3,vecangle_diff:3,vector:1,veri:3,versa:1,via:[0,2,3],vica:1,vid_board:[0,1],view:0,volt:[0,2],wai:[0,1],want:[1,3],web:[0,2,4],week:[0,1],well:[1,3],were:[0,1,2,3],what:[1,3],when:[0,1,2,3],where:[0,1,2,3],whether:1,which:[0,1,2,3,4,5],wide:3,width:3,window:[0,2,3],within:[0,1,2,3],without:[0,1],word:1,work:0,workhors:1,would:[0,1,2,3],wrapper:0,wsftneg:0,wspow08f3:0,wsvidalldn:0,xija:1,yaw:0,yaw_bias_diff:3,yaw_ctrl:3,yaw_stab:3,yaw_stab_perf:3,year:[0,1,2],yet:1,ylabel:3,ylim:3,you:[1,3],your:[1,3],yyyi:[0,2,3],zero:3,zone:[0,3]},titles:["API documentation","Commands and states","Event Descriptions","Chandra events","Kadi archive","Maneuver sequence templates"],titleterms:{"boolean":3,Use:3,aca:2,advanc:3,api:[0,4],archiv:[1,4],bad:[2,3],bsh_anom:5,calibr:2,cap:2,caveat:1,chandra:[1,3,4],combin:3,comm:2,command:[0,1,2,4],continu:1,current:2,dark:2,databas:2,date:1,defin:1,definit:3,delayed_npnt:5,descript:2,detail:3,doc:4,document:0,dsn:2,dump:2,dwell:2,eclips:2,event:[0,2,3,4],filter:3,four_ac:5,from:[1,2],get:3,grate:2,ground:2,help:3,hetg:2,identifi:2,ifot:2,implement:1,interfac:1,interm_att:5,interv:[2,3],kadi:4,kalman:2,kei:1,letg:2,line:1,load:2,ltt:[2,3],major:2,maneuv:[2,5],mode:2,model:0,momentum:2,more:3,movement:2,nman_dwel:5,non:3,normal:[2,5],note:1,nsun_anom:5,observ:2,obsid:3,orbit:2,over:1,overlap:3,overview:3,pad:3,pass:2,period:2,plan:2,point:2,queri:0,radiat:2,rang:1,relat:4,replica:2,run:2,safe:2,scs107:2,segment:2,select:3,sequenc:[2,5],sim:2,start:3,state:[0,1,4],sun:2,tabl:3,telemetri:3,templat:5,three_acq:5,three_acq_nman:5,time:3,translat:2,tsc:2,two_acq:5,two_acq_nman:5,user:1,zone:2}})