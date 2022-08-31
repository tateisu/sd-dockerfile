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

sub reLink($$){
	my($old,$new)=@_;

	(-e $old) or die "!! missing $old\n";
	($old eq $new) and die "!! symlink to same path. $new\n";

	if( -e $new ){
		cmd qq(rm -fr "$new");
	}
	unlink $new;
	symlink $old, $new;
	cmd qq(ls -lad $new);

	(-l $new) or die "!! can't symlink $new $!";
}

sub absPath($){
	my($old) = @_;
	return $old if $old =~ m|\A/|;
	return "/stable-diffusion/$old";
}


##################################

chdir "/stable-diffusion";
cmd qq(mkdir -p models/ldm/stable-diffusion-v1/);
cmd qq(mkdir -p host/outputs);
cmd qq(mkdir -p host/root-cache);

my $modelDst = "models/ldm/stable-diffusion-v1/model.ckpt";
my $modelFile = shift @ARGV || "host/models/sd-v1-1.ckpt";

reLink absPath($modelFile), $modelDst;
reLink absPath("host/outputs"), "outputs";
reLink absPath("host/root-cache"), "/root/.cache";

# cmd qq(nvidia-smi);
# cmd qq|python -c 'import torch; print(f"torch.cuda.is_available={torch.cuda.is_available()}")'|;
