#!/usr/bin/perl --
use strict;
use warnings;
use feature qw(say);

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
my $sd_repo = "https://github.com/basujindal/stable-diffusion";

my $build_context_dir = "docker-build-context";
my $sd_tgz = "$build_context_dir/stable-diffusion-src.tgz";
my $sd_src = "stable-diffusion-src";
if(not -f $sd_tgz ){
	if(not -d $sd_src){
		cmd qq(git clone --depth 1 $sd_repo $sd_src);
	}
	cmd qq(tar cpzf $sd_tgz -C $sd_src .);
}

cmd qq(docker image build -t tateisu/stable-diffusion:0.1 $build_context_dir);
