'''
Created Sept 28
@author: Jacob
'''
import json
import os
from subprocess import Popen, PIPE, STDOUT
import sys

'''
Observations:
Half of all retweet volume is from items only retweeted 1 or 2 times.
RT double's number of observed retweets. Most come with an @mention consistent with the mentionee being retweeted

What we want initially:
For each tweet, if a retweet store:
tweeterID, origID, origTweetID (if there), tweet text (if RT)
For now, we'll only look at the first three
'''

class SimpleTweetGraphMiner:
    '''
    classdocs
    '''


    def __init__(self,dStart,dEnd,hSkip):
        '''
        Constructor
        Loop that unzips files to memory then reads them.
        '''
        #For Zarya
        #unrarTool = "../../../../dfs/hulk/0/util/unrar"
        unrarTool = "/lfs/0/util/unrar"
	for d in range(dStart,dEnd):
            if d==dStart:
                hStart = hSkip
            for h in range(hStart,24):
                print d,h
                dstr = str(d)
                if d<10:
                    dstr = "0"+dstr
                hstr = str(h)
                if h<10:
                    hstr = "0"+hstr
                rarname = "/lfs/2/twitter-jan2011/twitter_201101"+dstr+"-"+hstr+".log.rar"
                tweetDataFile = open("/lfs/1/tmp/austin/tweetData"+dstr+"-"+hstr+".txt", "wb")
                unrarCMD = unrarTool+" p "+rarname+" "+"stdin"
                p=Popen(unrarCMD,shell=True,stdin=PIPE,stdout=PIPE,stderr=STDOUT,close_fds=True)
                unrarred = p.stdout
                i = 0
                for line in unrarred:
                    i+=1
        #            if i > 10000:
        #                break
                    if i%10000 == 0:
                        print i
                    tokens = line.split()
                    startChar = 0
                    #print line
                    try:
                        startChar = line.index("{")
                    except:
                        #print "No json"
                        pass
                    lineSub = line[startChar:]
                    info = {}
                    try:
                        info = json.loads(lineSub)
                        text = info["text"]
                        if "RT" in text:
                            #print "No RT"
                            continue
                    except:
                        #print "No text"
                        continue
                    #print info['text']
                    #print info['user']['lang']
                    if 'text' in info : #and 'entities' in info and 'user' in info :
                      #if 'user_mentions' in info['entities'] :
                      #  for mention in info['entities']['user_mentions'] :
                      try :
                            #tweetDataFile.write(info['user']['id_str'] + '\t' + mention['id_str'] + '\t' + info['text'] +'\r\n')
                          tweetDataFile.write(info['text'] +'\r\n')
                      except UnicodeEncodeError, e :
                        continue
                    '''
                    if info.has_key("retweeted_status"):
                        data = self.getFormalRTData(info)
                    else:
                        data = self.getInformalRTData(info)
                    rejsonnedData = json.dumps(data)
#                    print json.dumps(info,sort_keys=True, indent=4)
#                    print json.dumps(data,sort_keys=True, indent=4)
                    tweetDataFile.write(rejsonnedData+"\r\n")
                    ''' 
    def getFormalRTData(self,info):
        """
        Retrieve all sorts of tweet data from tweets with Retweeted_Status metadata
        getInformalRTData's schema
        +
        ["retweeted_status"] all of the above.
        """
        data = {}
        try:
            data = self.getInformalRTData(info)
            data["retweeted_status"] = self.getInformalRTData(info["retweeted_status"])
        except:
            data = {}
        return data
        
    def getInformalRTData(self,info):
        """
        ["created_at"]
        ["entities"]["hashtags"]
        ["entities"]["urls"]
        ["entities"]["user_mentions"]["id"]
        ["entities"]["user_mentions"]["screen_name"]
        ["id"]
        ["retweet_count"]
        ["user"]["followers_count"]
        ["user"]["friends_count"]
        ["user"]["lang"]
        ["user"]["listed_count"]
        ["user"]["screen_name"]
        ["user"]["statuses_count"]
        ["user"]["verified"]
        ["text"]
        """
        data = {}
        try:
            data["created_at"] = info["created_at"]
            data["entities"] = {}
            data["entities"]["hashtags"] = info["entities"]["hashtags"]
            data["entities"]["urls"] = info["entities"]["urls"]
            data["entities"]["user_mentions"] = info["entities"]["user_mentions"]
            data["id"] = info["id"] 
            data["retweet_count"] = info["retweet_count"]
            data["user"] = {}
            data["user"]["id"] = info["user"]["id"]
            data["user"]["followers_count"] = info["user"]["followers_count"]
            data["user"]["friends_count"] = info["user"]["friends_count"]
            data["user"]["lang"] = info["user"]["lang"]
            data["user"]["listed_count"] = info["user"]["listed_count"]
            data["user"]["screen_name"] = info["user"]["screen_name"]
            data["user"]["statuses_count"] = info["user"]["statuses_count"]
            data["user"]["verified"] = info["user"]["verified"] 
            data["text"] = info["text"]
        except:
            data = {}
        return data
   
def main():    
    SimpleTweetGraphMiner(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]))

if __name__=='__main__':
    main()
