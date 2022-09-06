#!/usr/bin/perl --
use strict;
use warnings;
use feature qw(say);
use Getopt::Long;

my $force =0;
GetOptions(
	"force:+" => \$force,
) or  die "bad options.\n";

sub cmd($){
	my($c)=@_;
	say $c;
	system $c;
	if( $? == 0 ){
		return 1;
	}elsif ($? == -1) {
        die "failed to execute: $!\n";
    }elsif ($? & 127) {
        die sprintf "child died with signal %d, %s coredump\n",
            ($? & 127),  ($? & 128) ? 'with' : 'without';
    }else {
        die sprintf "child exited with value %d\n", $? >> 8;
    }
}

##VRAMたりない my $sd_repo = "https://github.com/CompVis/stable-diffusion";
my $sd_repo = "https://github.com/tateisu/stable-diffusion";

my $build_context_dir = "docker-build-context";

my $sd_tgz = "$build_context_dir/stable-diffusion-src.tgz";
if($force or not -f $sd_tgz ){
	my $sd_src = "stable-diffusion-src";
	if($force or not -d $sd_src){
		cmd qq(rm -fr $sd_src);
		cmd qq(git clone --depth 1 $sd_repo $sd_src);
	}
	cmd qq(tar cpzf $sd_tgz -C $sd_src .);
}

my $re_tgz = "$build_context_dir/realesrgan.tgz";
if($force or not -f $re_tgz){
	my $zipfile = "realesrgan.zip";
	my $unzip_dir = "realesrgan";
	if($force or not -f $zipfile){
		cmd qq(rm -fr $zipfile);
		cmd qq(wget -O $zipfile https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip);
	}
	cmd qq(rm -fr $unzip_dir);
	cmd qq(mkdir -p $unzip_dir);
	cmd qq(unzip -o -d $unzip_dir $zipfile);
	cmd qq(rm -f $unzip_dir/*.jpg $unzip_dir/*.mp4);
	cmd qq(tar cpzf $re_tgz -C $unzip_dir .);
}

my @buildOptions = ();
if($force){
	push @buildOptions,"--no-cache";
}
my $opts = join ' ',@buildOptions;
cmd qq(docker image build $opts -t tateisu/stable-diffusion:0.1 $build_context_dir);
