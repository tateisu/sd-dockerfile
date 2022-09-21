#!/usr/bin/env perl
use strict;
use warnings;
use feature qw(say);
use utf8;

use List::Util qw(max);
use JSON; 
use HTML::Entities;

#########################

# 無視するキーのハッシュ
my %ignoreKeys = map{($_,1)} qw( C precision negative_prompt_alpha file );

# 表示時にキー名部分を出力しないキーのハッシュ
my %dontShowKeyName = map{($_,1)} qw( ckpt );

# キーの登場順序を覚えておく
my %keyOrder;
my $keyOrderSeed = 0;
$keyOrder{$_} = ++ $keyOrderSeed for (
    "ckpt",
    "width",
    "height",
    "sampler",
    "scale",
    "steps",
    "prompt",
    "negative_prompt",
    "Negative prompt",
    "Steps",
    "Sampler",
    "CFG scale",
    "Seed",
    "Size",
    "Denoising strength",
);

sub keyOrder($){
    $keyOrder{ $_[0] } or die "missing keyOrder for $_[0]\n";
}

#########################

# ファイルを読んでデータを返す
sub loadFile($;$){
    my($fName,$layer)=@_;
    $layer = "raw" if not $layer;
    open(my $fh,"<:$layer",$fName) or die "$fName $!";
    local $/=undef;
    my $text = <$fh>;
    close($fh) or die "$fName $!";
    return $text;
}

sub decodeAutomatic1111Text($){
    my @lines = split /\x0d?\x0a/, $_[0];
    my $info = {};
    my $lno=0;
    for(@lines){
        ++$lno;
        if($lno==1){
            $info->{prompt}=$_;
            next;
        }elsif( /^Negative prompt: (.*)/){
            $info->{negative_prompt}=$1;
            next;
        }
        for(split /, /,$_){
            my($k,$v) = split /: /,$_;
            if(defined $k and defined $v){
                $info->{$k}=$v;
            }
        }
    }
    return $info;
}

# DATAを読んで grid を作る
my @grid = ();
my $curRow;
sub addRow{
    $curRow = [];
    push @grid,$curRow;
}
sub addInfo{
    my($item,$info)=@_;
    while( my($k,$v)= each %$info ){
        $item->{$k}=$v;
        if(not defined $keyOrder{$k}){
            $keyOrder{$k} = ++ $keyOrderSeed;
        }
    }
}
while(<DATA>){
    s|//.+||g;
    s/[\x0d\x0a]+//g;
    next if not length;
    if( /^\s*ROW\s*$/){
        addRow();
        next;
    }
    addRow() if not $curRow;
    die "missing file. $_ " if not -f $_;
    my $item = { file=>$_};
    push @$curRow,$item;

    my $infoFile = $_;
    $infoFile =~ s/\.png|\.x4-.+/_info.txt/;
    if( -f $infoFile){
        my $info = decode_json loadFile($infoFile);
        addInfo($item,$info);
    }else{
        $infoFile = $_;
        $infoFile =~ s/\.png|\.x4-.+/.txt/;
        if( -f $infoFile){
            my $info = decodeAutomatic1111Text loadFile($infoFile,"utf8");
            addInfo($item,$info);
        }
    }
}

# グリッドのサイズ
my $gridWidth = max( map{ 0+ @$_} @grid);
my $gridHeight = 0+@grid;

##########################################

my @topHeaders;
my @leftHeaders;
my %commonHeader;

# 上端ヘッダ、左端ヘッダ、共通ヘッダにkey,valueを追加する
sub addHeaderValue{
    if(@_==4){
        my($k,$v,$headers,$idx)=@_;
        my $header = $headers->[$idx];
        $header or $header = $headers->[$idx] = {};
        my $values = $header->{$k};
        $values or $values = $header->{$k} = {};
        $values->{$v} =1;
    }else{
        my($k,$v,$header)=@_;
        my $values = $header->{$k};
        $values or $values = $header->{$k} = {};
        $values->{$v} =1;
    }
}
for( my $iy=0;$iy<$gridHeight;++$iy){
    for( my $ix=0;$ix<$gridWidth;++$ix){
        my $item = $grid[$iy][$ix] or next;
        while( my($k,$v)=each %$item){
            addHeaderValue($k,$v,\@topHeaders,$ix);
            addHeaderValue($k,$v,\@leftHeaders,$iy);
            addHeaderValue($k,$v,\%commonHeader);
        }
    }
}

# ヘッダ中のkeyごとに、valueが同一ならその項目を残し、そうでなければキーを削除する
sub fixHeader{
    my($header)=@_;
    for my $key (keys %$header){
        my @values = keys %{$header->{$key}};
        if(@values>1){
            delete $header->{$key};
        }else{
            $header->{$key} = $values[0];
        }
    }
}
fixHeader($_) for @topHeaders;
fixHeader($_) for @leftHeaders;
fixHeader(\%commonHeader);

# 共通ヘッダにデータがあるものは上端ヘッダ、左端ヘッダから削除する
for my $key (keys %commonHeader){
    delete $_->{$key} for @topHeaders;
    delete $_->{$key} for @leftHeaders;
}

# 上端ヘッダ、左端ヘッダを表示するかどうか
my $hasTopHeader = max( map{ 0+keys(%$_)} @topHeaders );
my $hasLeftHeader = max( map{ 0+keys(%$_)} @leftHeaders );

##########################################
# HTML出力

# ヘッダの内容をHTMLのリストとして出力する
sub listHeader{
    my($header,$title)=@_;

    my @keys = sort { keyOrder($a) <=> keyOrder($b) } 
        grep{!$ignoreKeys{$_}} 
        keys %$header;

    return if not @keys;

    if($title){
        say qq(<H3>),encode_entities($title),qq(</H3>);
    }
    say qq(<ul>);
    for my $key ( @keys ){
        say qq(<li>);
        if($dontShowKeyName{$key}){
            say encode_entities($header->{$key});
        }else{
            say encode_entities($key),": ",encode_entities($header->{$key});
        }
        say qq(</li>);
    }
    say qq(</ul>);
}

say <<"END";
<html lang="ja">
<style>
img.box{
    width: 256px;
    height: auto;
}
ul {
  list-style: none;
  padding-left: 0;
  font-size: 60%;
}
caption{
    text-align: start;
}
</style>
<table><caption>
END

listHeader(\%commonHeader);

say <<"END";
</caption>
END

if($hasTopHeader){
    say qq(<thead><tr>);
    if( $hasLeftHeader ){
        say qq(<td>&nbsp;</td>);
    }
    for my $header(@topHeaders){
        say qq(<th>);
        listHeader($header);
        say qq(</th>);
    }
    say qq(</tr></thead>);
}

for( my $iy=0;$iy<$gridHeight;++$iy){
    my $row = $grid[$iy];
    say qq(<tr>);
    
    if($hasLeftHeader){
        say qq(<th>);
        listHeader($leftHeaders[$iy]);
        say qq(</th>);
    }

    for( my $ix=0;$ix<$gridWidth;++$ix){
        my $item = $row->[$ix];
        say qq(<td>);
        say qq(<img class="box" src="$item->{file}"/>);
        my %cellHeader = map{ ($_,$item->{$_}) } grep{
            $_ ne "file"
            and not $commonHeader{$_}
            and not $topHeaders[$ix]{$_}
            and not $leftHeaders[$iy]{$_}
        } keys %$item;
        listHeader(\%cellHeader);
        say qq(</td>);
    }
    say qq(</tr>);
}
say <<"END";
</table>
</html>
END

__DATA__

outputs/20220920231656_1535587772.png
outputs/20220920232008_2480661817.png
outputs/20220920232313_1730805163.png
outputs/20220920232618_1730805164.png

outputs/20220920232923_1730805165.png
outputs/20220920233227_1730805166.png
outputs/20220920233523_1730805167.png
outputs/20220920233818_1730805168.png

outputs/20220920234114_1730805169.png
outputs/20220920234409_1730805170.png
outputs/20220920234705_1730805171.png
outputs/20220920235001_1730805172.png
